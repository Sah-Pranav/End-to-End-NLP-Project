from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from summarization.config import (
    load_generation_config,
    load_tokenization_config,
    load_yaml,
)
from summarization.inference import summarize
from summarization.logging_utils import (
    configure_logging,
    get_logger,
)
from summarization.models import load_fine_tuned_model
from summarization.schemas import (
    SummarizationRequest,
    SummarizationResponse,
)


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference resources when the application starts."""

    configure_logging()

    logger.info("Starting summarization API.")

    config = load_yaml("config/config.yaml")

    adapter_path = Path(
        config["model_artifacts"]["fine_tuned_dir"]
    )

    logger.info(
        "Loading fine-tuned model from: %s",
        adapter_path,
    )

    app.state.loaded_model = load_fine_tuned_model(
        adapter_path,
    )

    app.state.generation_config = (
        load_generation_config()
    )

    app.state.tokenization_config = (
        load_tokenization_config()
    )

    logger.info(
        "Summarization API is ready. Model: %s",
        app.state.loaded_model.model_name,
    )

    yield

    logger.info("Shutting down summarization API.")


app = FastAPI(
    title="Text and Dialogue Summarization API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": (
            "Text and Dialogue Summarization API"
        )
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "healthy",
        "model_name": app.state.loaded_model.model_name,
    }


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Return whether the application is ready for inference."""

    return {
        "status": "ready",
        "model_name": app.state.loaded_model.model_name,
    }


@app.post(
    "/summarize",
    response_model=SummarizationResponse,
)
def summarize_text(
    request: SummarizationRequest,
) -> SummarizationResponse:
    """Generate a summary for the provided text."""

    logger.info(
        "Received summarization request: "
        "input_length=%d characters",
        len(request.text),
    )

    try:
        result = summarize(
            loaded_model=app.state.loaded_model,
            text=request.text,
            generation_config=(
                app.state.generation_config
            ),
            tokenization_config=(
                app.state.tokenization_config
            ),
        )
    except Exception:
        logger.exception(
            "Summarization inference failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Failed to generate summary.",
        )

    logger.info(
        "Summarization completed: "
        "latency_ms=%.2f, summary_length=%d characters",
        result.latency_ms,
        len(result.summary),
    )

    return SummarizationResponse(
        summary=result.summary,
        model_name=(
            app.state.loaded_model.model_name
        ),
        latency_ms=result.latency_ms,
    )
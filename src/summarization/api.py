from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from summarization.config import (
    load_generation_config,
    load_tokenization_config,
    load_yaml,
)
from summarization.inference import summarize
from summarization.models import load_fine_tuned_model
from summarization.schemas import (
    SummarizationRequest,
    SummarizationResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference resources when the application starts."""

    config = load_yaml("config/config.yaml")

    adapter_path = Path(
        config["model_artifacts"]["fine_tuned_dir"]
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

    yield


app = FastAPI(
    title="Text and Dialogue Summarization API",
    version="0.1.0",
    lifespan=lifespan,
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

    return SummarizationResponse(
        summary=result.summary,
        model_name=(
            app.state.loaded_model.model_name
        ),
        latency_ms=result.latency_ms,
    )

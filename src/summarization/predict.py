from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from summarization.config import (
    load_generation_config,
    load_model_config,
    load_tokenization_config,
    load_yaml,
)
from summarization.inference import summarize
from summarization.logging_utils import (
    configure_logging,
    get_logger,
)
from summarization.models import (
    LoadedModel,
    load_fine_tuned_model,
    load_model,
)


logger = get_logger(__name__)


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for summarization inference."""

    parser = argparse.ArgumentParser(
        description="Generate a summary for custom input text."
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text to summarize.",
    )

    parser.add_argument(
        "--model",
        choices=(
            "fine-tuned",
            "efficient-baseline",
            "reference",
        ),
        default="fine-tuned",
        help=(
            "Model to use for inference. "
            "Default: fine-tuned."
        ),
    )

    return parser.parse_args(arguments)


def load_selected_model(
    model_type: str,
) -> LoadedModel:
    """Load the model selected for inference."""

    model_config = load_model_config()

    if model_type == "efficient-baseline":
        logger.info(
            "Loading efficient baseline model: %s",
            model_config.efficient_baseline,
        )

        return load_model(
            model_config.efficient_baseline,
        )

    if model_type == "reference":
        logger.info(
            "Loading reference model: %s",
            model_config.reference,
        )

        return load_model(
            model_config.reference,
        )

    config = load_yaml("config/config.yaml")

    adapter_path = Path(
        config["model_artifacts"]["fine_tuned_dir"]
    )

    if not adapter_path.exists():
        raise FileNotFoundError(
            "Fine-tuned model adapter was not found at: "
            f"{adapter_path}"
        )

    logger.info(
        "Loading fine-tuned model from: %s",
        adapter_path,
    )

    return load_fine_tuned_model(
        adapter_path,
    )


def main() -> None:
    """Run the command-line summarization demonstration."""

    configure_logging()

    arguments = parse_arguments()

    generation_config = load_generation_config()
    tokenization_config = load_tokenization_config()

    loaded_model = load_selected_model(
        arguments.model,
    )

    result = summarize(
        loaded_model=loaded_model,
        text=arguments.text,
        generation_config=generation_config,
        tokenization_config=tokenization_config,
    )

    print()
    print(f"Model: {loaded_model.model_name}")
    print()
    print("Summary:")
    print(result.summary)
    print()
    print(
        "Inference latency: "
        f"{result.latency_ms:.2f} ms"
    )
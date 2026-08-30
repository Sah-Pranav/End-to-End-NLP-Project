from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from summarization.benchmark import benchmark_model
from summarization.config import (
    EvaluationConfig,
    GenerationConfig,
    TokenizationConfig,
    load_data_config,
    load_evaluation_config,
    load_generation_config,
    load_model_config,
    load_results_config,
    load_tokenization_config,
    load_yaml,
)
from summarization.data import load_xsum
from summarization.logging_utils import (
    configure_logging,
    get_logger,
)
from summarization.models import (
    LoadedModel,
    load_efficient_baseline,
    load_fine_tuned_model,
    load_model,
)
from summarization.results import save_benchmark_result


logger = get_logger(__name__)


def benchmark_and_save(
    dataset: Dataset,
    loaded_model: LoadedModel,
    generation_config: GenerationConfig,
    tokenization_config: TokenizationConfig,
    evaluation_config: EvaluationConfig,
    output_path: str,
) -> None:
    """Benchmark a model and save its results."""

    result = benchmark_model(
        dataset=dataset,
        loaded_model=loaded_model,
        generation_config=generation_config,
        tokenization_config=tokenization_config,
        evaluation_config=evaluation_config,
    )

    save_benchmark_result(
        result=result,
        output_path=output_path,
    )


def main() -> None:
    """Benchmark baseline, base, and fine-tuned models."""

    configure_logging()

    logger.info(
        "Starting model comparison benchmark."
    )

    data_config = load_data_config()
    model_config = load_model_config()
    generation_config = load_generation_config()
    tokenization_config = load_tokenization_config()
    evaluation_config = load_evaluation_config()
    results_config = load_results_config()

    project_config = load_yaml(
        "config/config.yaml"
    )

    adapter_path = Path(
        project_config["model_artifacts"]["fine_tuned_dir"]
    )

    logger.info(
        "Loading dataset: %s",
        data_config.dataset_name,
    )

    dataset = load_xsum(data_config)
    test_dataset = dataset["test"]

    logger.info(
        "Test dataset ready: examples=%d",
        len(test_dataset),
    )

    logger.info(
        "Benchmarking efficient baseline."
    )

    efficient_baseline = load_efficient_baseline(
        model_config,
    )

    benchmark_and_save(
        dataset=test_dataset,
        loaded_model=efficient_baseline,
        generation_config=generation_config,
        tokenization_config=tokenization_config,
        evaluation_config=evaluation_config,
        output_path=results_config.metrics_file,
    )

    logger.info(
        "Benchmarking base model."
    )

    base_model = load_model(
        model_config.primary,
    )

    benchmark_and_save(
        dataset=test_dataset,
        loaded_model=base_model,
        generation_config=generation_config,
        tokenization_config=tokenization_config,
        evaluation_config=evaluation_config,
        output_path=results_config.metrics_file,
    )

    logger.info(
        "Benchmarking fine-tuned LoRA model."
    )

    fine_tuned_model = load_fine_tuned_model(
        adapter_path,
    )

    benchmark_and_save(
        dataset=test_dataset,
        loaded_model=fine_tuned_model,
        generation_config=generation_config,
        tokenization_config=tokenization_config,
        evaluation_config=evaluation_config,
        output_path=results_config.metrics_file,
    )

    logger.info(
        "Model comparison benchmark completed."
    )


if __name__ == "__main__":
    main()
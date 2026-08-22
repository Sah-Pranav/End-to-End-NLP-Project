from __future__ import annotations

import statistics
from dataclasses import dataclass

from datasets import Dataset

from summarization.config import (
    EvaluationConfig,
    GenerationConfig,
    TokenizationConfig,
)
from summarization.evaluation import (
    EvaluationResult,
    evaluate_summaries,
)
from summarization.inference import summarize
from summarization.logging_utils import get_logger
from summarization.models import LoadedModel


logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from benchmarking a summarization model."""

    model_name: str
    num_examples: int
    evaluation: EvaluationResult
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float


def calculate_latency_statistics(
    latencies_ms: list[float],
) -> tuple[float, float, float]:
    """Calculate mean, median, and P95 latency."""

    if not latencies_ms:
        raise ValueError("Latency measurements must not be empty.")

    mean_latency = statistics.mean(latencies_ms)
    median_latency = statistics.median(latencies_ms)

    sorted_latencies = sorted(latencies_ms)

    p95_index = min(
        int(len(sorted_latencies) * 0.95),
        len(sorted_latencies) - 1,
    )

    p95_latency = sorted_latencies[p95_index]

    return (
        float(mean_latency),
        float(median_latency),
        float(p95_latency),
    )


def benchmark_model(
    dataset: Dataset,
    loaded_model: LoadedModel,
    generation_config: GenerationConfig,
    tokenization_config: TokenizationConfig,
    evaluation_config: EvaluationConfig,
) -> BenchmarkResult:
    """Benchmark a summarization model on a dataset."""

    if len(dataset) == 0:
        raise ValueError("Benchmark dataset must not be empty.")

    max_samples = evaluation_config.max_samples

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")

        num_examples = min(max_samples, len(dataset))
        dataset = dataset.select(range(num_examples))
    else:
        num_examples = len(dataset)

    logger.info(
        "Benchmark started: model=%s examples=%d",
        loaded_model.model_name,
        num_examples,
    )

    predictions: list[str] = []
    reference_summaries: list[str] = []
    latencies_ms: list[float] = []

    documents = dataset["document"]
    summaries = dataset["summary"]

    for document, reference in zip(documents, summaries):
        result = summarize(
            loaded_model=loaded_model,
            text=str(document),
            generation_config=generation_config,
            tokenization_config=tokenization_config,
        )

        predictions.append(result.summary)
        reference_summaries.append(str(reference))
        latencies_ms.append(result.latency_ms)

    evaluation = evaluate_summaries(
        predictions=predictions,
        references=reference_summaries,
    )

    mean_latency, median_latency, p95_latency = (
        calculate_latency_statistics(latencies_ms)
    )

    result = BenchmarkResult(
        model_name=loaded_model.model_name,
        num_examples=num_examples,
        evaluation=evaluation,
        mean_latency_ms=mean_latency,
        median_latency_ms=median_latency,
        p95_latency_ms=p95_latency,
    )

    logger.info(
        "Benchmark completed: model=%s examples=%d",
        result.model_name,
        result.num_examples,
    )

    return result
import json

from summarization.benchmark import BenchmarkResult
from summarization.evaluation import EvaluationResult
from summarization.results import (
    load_benchmark_results,
    save_benchmark_result,
)


def create_benchmark_result(
    model_name: str = "test-model",
    rouge1: float = 0.21,
) -> BenchmarkResult:
    """Create a benchmark result for testing."""

    return BenchmarkResult(
        model_name=model_name,
        num_examples=20,
        evaluation=EvaluationResult(
            rouge1=rouge1,
            rouge2=0.05,
            rougeL=0.14,
        ),
        mean_latency_ms=1800.0,
        median_latency_ms=1700.0,
        p95_latency_ms=2400.0,
    )


def test_save_benchmark_result_creates_file(tmp_path):
    result = create_benchmark_result()

    output_path = tmp_path / "results" / "benchmark.json"

    save_benchmark_result(
        result=result,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert len(data["models"]) == 1
    assert data["models"][0]["model_name"] == "test-model"
    assert data["models"][0]["num_examples"] == 20
    assert data["models"][0]["evaluation"]["rouge1"] == 0.21
    assert data["models"][0]["evaluation"]["rouge2"] == 0.05
    assert data["models"][0]["evaluation"]["rougeL"] == 0.14
    assert data["models"][0]["mean_latency_ms"] == 1800.0


def test_save_benchmark_result_preserves_multiple_models(
    tmp_path,
):
    output_path = tmp_path / "results" / "benchmark.json"

    first_result = create_benchmark_result(
        model_name="model-one",
    )
    second_result = create_benchmark_result(
        model_name="model-two",
    )

    save_benchmark_result(
        result=first_result,
        output_path=output_path,
    )
    save_benchmark_result(
        result=second_result,
        output_path=output_path,
    )

    results = load_benchmark_results(
        output_path,
    )

    assert len(results["models"]) == 2
    assert results["models"][0]["model_name"] == "model-one"
    assert results["models"][1]["model_name"] == "model-two"


def test_save_benchmark_result_updates_existing_model(
    tmp_path,
):
    output_path = tmp_path / "results" / "benchmark.json"

    first_result = create_benchmark_result(
        model_name="test-model",
        rouge1=0.21,
    )
    updated_result = create_benchmark_result(
        model_name="test-model",
        rouge1=0.35,
    )

    save_benchmark_result(
        result=first_result,
        output_path=output_path,
    )
    save_benchmark_result(
        result=updated_result,
        output_path=output_path,
    )

    results = load_benchmark_results(
        output_path,
    )

    assert len(results["models"]) == 1
    assert (
        results["models"][0]["evaluation"]["rouge1"]
        == 0.35
    )


def test_load_benchmark_results_migrates_legacy_format(
    tmp_path,
):
    output_path = tmp_path / "benchmark.json"

    legacy_data = {
        "model_name": "legacy-model",
        "num_examples": 20,
        "evaluation": {
            "rouge1": 0.21,
            "rouge2": 0.05,
            "rougeL": 0.14,
        },
        "mean_latency_ms": 1800.0,
        "median_latency_ms": 1700.0,
        "p95_latency_ms": 2400.0,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(legacy_data, file)

    results = load_benchmark_results(
        output_path,
    )

    assert len(results["models"]) == 1
    assert (
        results["models"][0]["model_name"]
        == "legacy-model"
    )


def test_load_benchmark_results_returns_empty_for_missing_file(
    tmp_path,
):
    output_path = tmp_path / "missing.json"

    results = load_benchmark_results(
        output_path,
    )

    assert results == {"models": []}
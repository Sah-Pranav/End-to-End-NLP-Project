import json

from summarization.benchmark import BenchmarkResult
from summarization.evaluation import EvaluationResult
from summarization.results import save_benchmark_result


def test_save_benchmark_result(tmp_path):
    result = BenchmarkResult(
        model_name="test-model",
        num_examples=20,
        evaluation=EvaluationResult(
            rouge1=0.21,
            rouge2=0.05,
            rougeL=0.14,
        ),
        mean_latency_ms=1800.0,
        median_latency_ms=1700.0,
        p95_latency_ms=2400.0,
    )

    output_path = tmp_path / "results" / "benchmark.json"

    save_benchmark_result(
        result=result,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["model_name"] == "test-model"
    assert data["num_examples"] == 20
    assert data["evaluation"]["rouge1"] == 0.21
    assert data["evaluation"]["rouge2"] == 0.05
    assert data["evaluation"]["rougeL"] == 0.14
    assert data["mean_latency_ms"] == 1800.0

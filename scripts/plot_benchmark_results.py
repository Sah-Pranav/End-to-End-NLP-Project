from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/model_comparison.json")
OUTPUT_DIR = Path("docs/results")


def load_results(
    results_path: Path,
) -> list[dict]:
    """Load model comparison benchmark results."""

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["models"]


def create_rouge_chart(
    results: list[dict],
    output_path: Path,
) -> None:
    """Create a ROUGE score comparison chart."""

    model_names = [
        result["model_name"]
        for result in results
    ]

    rouge1_scores = [
        result["evaluation"]["rouge1"]
        for result in results
    ]

    rouge2_scores = [
        result["evaluation"]["rouge2"]
        for result in results
    ]

    rougeL_scores = [
        result["evaluation"]["rougeL"]
        for result in results
    ]

    positions = list(range(len(model_names)))
    width = 0.25

    plt.figure(figsize=(10, 7))

    plt.bar(
        [position - width for position in positions],
        rouge1_scores,
        width=width,
        label="ROUGE-1",
    )

    plt.bar(
        positions,
        rouge2_scores,
        width=width,
        label="ROUGE-2",
    )

    plt.bar(
        [position + width for position in positions],
        rougeL_scores,
        width=width,
        label="ROUGE-L",
    )

    plt.xticks(
        positions,
        model_names,
        rotation=0,
        ha="center",
    )

    plt.ylabel("ROUGE Score")
    plt.title("Summarization Model Comparison")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def create_latency_chart(
    results: list[dict],
    output_path: Path,
) -> None:
    """Create a mean latency comparison chart."""

    model_names = [
        result["model_name"]
        for result in results
    ]

    mean_latencies = [
        result["mean_latency_ms"]
        for result in results
    ]

    plt.figure(figsize=(10, 7))

    plt.bar(
        model_names,
        mean_latencies,
    )

    plt.xticks(
        rotation=0,
        ha="center",
    )

    plt.ylabel("Mean Latency (ms)")
    plt.title("Summarization Model Inference Latency")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main() -> None:
    """Generate benchmark result visualizations."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results(
        RESULTS_PATH,
    )

    create_rouge_chart(
        results=results,
        output_path=OUTPUT_DIR
        / "rouge_comparison.png",
    )

    create_latency_chart(
        results=results,
        output_path=OUTPUT_DIR
        / "latency_comparison.png",
    )

    print(
        "Benchmark visualizations saved to "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
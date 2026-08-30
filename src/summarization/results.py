from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from summarization.benchmark import BenchmarkResult
from summarization.logging_utils import get_logger


logger = get_logger(__name__)


def save_benchmark_result(
    result: BenchmarkResult,
    output_path: str | Path,
) -> None:
    """Save or update a benchmark result in a comparison JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Saving benchmark result: %s",
        output_path,
    )

    results = load_benchmark_results(
        output_path,
    )

    result_payload = asdict(result)

    for index, existing_result in enumerate(
        results["models"]
    ):
        if (
            existing_result.get("model_name")
            == result.model_name
        ):
            results["models"][index] = result_payload
            break
    else:
        results["models"].append(
            result_payload
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )
        file.write("\n")

    logger.info(
        "Benchmark result saved: %s",
        output_path,
    )


def load_benchmark_results(
    output_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load benchmark comparison results from JSON."""

    output_path = Path(output_path)

    if not output_path.exists():
        return {"models": []}

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Benchmark results must contain "
            "a JSON object."
        )

    if "models" not in data:
        if "model_name" in data:
            return {"models": [data]}

        raise ValueError(
            "Benchmark results must contain "
            "a 'models' field."
        )

    if not isinstance(data["models"], list):
        raise ValueError(
            "Benchmark results 'models' field "
            "must be a list."
        )

    return data
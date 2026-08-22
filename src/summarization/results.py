from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from summarization.benchmark import BenchmarkResult
from summarization.logging_utils import get_logger


logger = get_logger(__name__)


def save_benchmark_result(
    result: BenchmarkResult,
    output_path: str | Path,
) -> None:
    """Save a benchmark result as structured JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving benchmark result: %s", output_path)

    payload = asdict(result)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    logger.info("Benchmark result saved: %s", output_path)

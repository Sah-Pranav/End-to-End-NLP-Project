from __future__ import annotations

from dataclasses import dataclass

import evaluate


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for generated summaries."""

    rouge1: float
    rouge2: float
    rougeL: float


def evaluate_summaries(
    predictions: list[str],
    references: list[str],
) -> EvaluationResult:
    """Calculate ROUGE metrics for generated summaries."""

    if not predictions:
        raise ValueError("Predictions must not be empty.")

    if not references:
        raise ValueError("References must not be empty.")

    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must contain the same "
            "number of examples."
        )

    rouge = evaluate.load("rouge")

    scores = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return EvaluationResult(
        rouge1=float(scores["rouge1"]),
        rouge2=float(scores["rouge2"]),
        rougeL=float(scores["rougeL"]),
    )

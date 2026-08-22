import pytest

from summarization.evaluation import (
    EvaluationResult,
    evaluate_summaries,
)


def test_evaluation_returns_rouge_scores():
    predictions = [
        "The cat is sitting on the mat.",
        "The weather is sunny today.",
    ]

    references = [
        "A cat sits on the mat.",
        "Today the weather is sunny.",
    ]

    result = evaluate_summaries(
        predictions=predictions,
        references=references,
    )

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.rouge1 <= 1.0
    assert 0.0 <= result.rouge2 <= 1.0
    assert 0.0 <= result.rougeL <= 1.0


def test_evaluation_rejects_empty_predictions():
    with pytest.raises(ValueError, match="Predictions must not be empty"):
        evaluate_summaries(
            predictions=[],
            references=["A reference summary."],
        )


def test_evaluation_rejects_empty_references():
    with pytest.raises(ValueError, match="References must not be empty"):
        evaluate_summaries(
            predictions=["A generated summary."],
            references=[],
        )


def test_evaluation_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="same number of examples",
    ):
        evaluate_summaries(
            predictions=["Summary one.", "Summary two."],
            references=["Reference one."],
        )

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

from summarization.config import TokenizationConfig
from summarization.tokenization import tokenize_dataset


class MockTokenizer:
    """Minimal tokenizer implementation for unit tests."""

    def __call__(
        self,
        text: Any = None,
        *,
        text_target: Any = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        del truncation, kwargs

        values = text_target if text_target is not None else text

        if values is None:
            raise ValueError("Tokenizer input must not be None.")

        input_ids: list[list[int]] = []

        for value in values:
            tokens = list(
                range(
                    min(
                        len(value.split()),
                        (
                            max_length
                            if max_length is not None
                            else len(value.split())
                        ),
                    )
                )
            )
            input_ids.append(tokens)

        result: dict[str, list[list[int]]] = {
            "input_ids": input_ids,
        }

        if text_target is None:
            result["attention_mask"] = [
                [1] * len(tokens)
                for tokens in input_ids
            ]

        return result


@pytest.fixture
def tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        max_input_length=5,
        max_target_length=3,
    )


def test_tokenize_dataset_creates_expected_columns(
    tokenization_config: TokenizationConfig,
):
    dataset = Dataset.from_dict(
        {
            "document": ["one two three"],
            "summary": ["one two"],
        }
    )

    result = tokenize_dataset(
        dataset=dataset,
        tokenizer=MockTokenizer(),
        config=tokenization_config,
    )

    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names
    assert "labels" in result.column_names


def test_tokenize_dataset_respects_length_limits(
    tokenization_config: TokenizationConfig,
):
    dataset = Dataset.from_dict(
        {
            "document": [
                "one two three four five six seven"
            ],
            "summary": [
                "one two three four five"
            ],
        }
    )

    result = tokenize_dataset(
        dataset=dataset,
        tokenizer=MockTokenizer(),
        config=tokenization_config,
    )

    assert len(result["input_ids"][0]) <= 5
    assert len(result["labels"][0]) <= 3


def test_tokenize_dataset_removes_original_columns(
    tokenization_config: TokenizationConfig,
):
    dataset = Dataset.from_dict(
        {
            "document": ["one two"],
            "summary": ["one"],
        }
    )

    result = tokenize_dataset(
        dataset=dataset,
        tokenizer=MockTokenizer(),
        config=tokenization_config,
    )

    assert "document" not in result.column_names
    assert "summary" not in result.column_names


def test_tokenize_dataset_rejects_missing_columns(
    tokenization_config: TokenizationConfig,
):
    dataset = Dataset.from_dict(
        {
            "document": ["one two"],
        }
    )

    with pytest.raises(ValueError, match="summary"):
        tokenize_dataset(
            dataset=dataset,
            tokenizer=MockTokenizer(),
            config=tokenization_config,
        )


def test_tokenize_dataset_rejects_empty_dataset(
    tokenization_config: TokenizationConfig,
):
    dataset = Dataset.from_dict(
        {
            "document": [],
            "summary": [],
        }
    )

    with pytest.raises(ValueError, match="must not be empty"):
        tokenize_dataset(
            dataset=dataset,
            tokenizer=MockTokenizer(),
            config=tokenization_config,
        )
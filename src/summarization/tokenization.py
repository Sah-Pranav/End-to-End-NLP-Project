from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from datasets import Dataset

from summarization.config import TokenizationConfig
from summarization.logging_utils import get_logger


REQUIRED_COLUMNS = {"document", "summary"}

logger = get_logger(__name__)


class TokenizerProtocol(Protocol):
    """Interface required by the tokenization layer."""

    def __call__(
        self,
        text: Any = None,
        *,
        text_target: Any = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        ...


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: TokenizerProtocol,
    config: TokenizationConfig,
) -> Dataset:
    """Tokenize documents and reference summaries for seq2seq training."""

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataset) == 0:
        raise ValueError("Dataset must not be empty.")

    logger.info(
        "Tokenizing dataset: examples=%d input_length=%d target_length=%d",
        len(dataset),
        config.max_input_length,
        config.max_target_length,
    )

    def tokenize_batch(
        examples: dict[str, list[str]],
    ) -> dict[str, Any]:
        model_inputs = tokenizer(
            examples["document"],
            max_length=config.max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=examples["summary"],
            max_length=config.max_target_length,
            truncation=True,
        )

        tokenized_inputs = dict(model_inputs)
        tokenized_inputs["labels"] = labels["input_ids"]

        return tokenized_inputs

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    logger.info(
        "Dataset tokenization completed: examples=%d",
        len(tokenized_dataset),
    )

    return tokenized_dataset
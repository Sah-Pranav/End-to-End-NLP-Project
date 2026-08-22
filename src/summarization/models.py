from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from summarization.config import ModelConfig

from summarization.logging_utils import get_logger

logger = get_logger(__name__)


class Seq2SeqModel(Protocol):
    """Interface required by the inference layer."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    def generate(self, **kwargs: Any) -> torch.Tensor:
        ...


@dataclass
class LoadedModel:
    """Container for a tokenizer and its corresponding model."""

    tokenizer: PreTrainedTokenizerBase
    model: Seq2SeqModel
    model_name: str


def load_model(model_name: str) -> LoadedModel:
    """Load a pretrained sequence-to-sequence model and tokenizer."""

    logger.info("Loading model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info("Model loaded: %s", model_name)

    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
    )


def load_efficient_baseline(
    config: ModelConfig,
) -> LoadedModel:
    """Load the configured efficient baseline model."""

    return load_model(config.efficient_baseline)
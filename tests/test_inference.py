import pytest
import torch

from summarization.config import (
    GenerationConfig,
    TokenizationConfig,
)
from summarization.inference import summarize
from summarization.models import LoadedModel


class FakeTokenizer:
    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_tensors,
    ):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens):
        return "generated summary"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def generate(self, **kwargs):
        return torch.tensor([[4, 5, 6]])


def create_loaded_model() -> LoadedModel:
    return LoadedModel(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_name="fake-model",
    )


def create_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=64,
        num_beams=4,
        do_sample=False,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
    )


def create_tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        max_input_length=512,
        max_target_length=64,
    )


def test_summarize_returns_summary_and_latency():
    result = summarize(
        loaded_model=create_loaded_model(),
        text="This is a test document.",
        generation_config=create_generation_config(),
        tokenization_config=create_tokenization_config(),
    )

    assert result.summary == "generated summary"
    assert result.latency_ms >= 0


def test_summarize_rejects_empty_text():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize(
            loaded_model=create_loaded_model(),
            text="   ",
            generation_config=create_generation_config(),
            tokenization_config=create_tokenization_config(),
        )

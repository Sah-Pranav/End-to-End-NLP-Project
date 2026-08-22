from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from summarization.config import GenerationConfig, TokenizationConfig
from summarization.logging_utils import get_logger
from summarization.models import LoadedModel


logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Result returned by the summarization inference pipeline."""

    summary: str
    latency_ms: float


def summarize(
    loaded_model: LoadedModel,
    text: str,
    generation_config: GenerationConfig,
    tokenization_config: TokenizationConfig,
) -> InferenceResult:
    """Generate a summary for a single input document."""

    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    logger.debug("Inference started")

    model = loaded_model.model
    tokenizer = loaded_model.tokenizer

    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        max_length=tokenization_config.max_input_length,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            max_length=None,
            num_beams=generation_config.num_beams,
            do_sample=generation_config.do_sample,
            length_penalty=generation_config.length_penalty,
            no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
        )

    if device.type == "mps":
        torch.mps.synchronize()

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.debug(
        "Inference completed: latency_ms=%.2f",
        latency_ms,
    )

    decoded = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    if not isinstance(decoded, str):
        raise TypeError("Tokenizer decode must return a string.")

    summary = decoded.strip()

    return InferenceResult(
        summary=summary,
        latency_ms=latency_ms,
    )
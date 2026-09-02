from typing import cast

import pytest
from fastapi.testclient import TestClient

from summarization import api
from summarization.config import (
    GenerationConfig,
    TokenizationConfig,
)
from summarization.inference import InferenceResult
from summarization.models import (
    LoadedModel,
    Seq2SeqModel,
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)


class FakeTokenizer:
    pass


class FakeModel:
    pass


@pytest.fixture
def client(monkeypatch):
    loaded_model = LoadedModel(
        tokenizer=cast(
            PreTrainedTokenizerBase,
            FakeTokenizer(),
        ),
        model=cast(
            Seq2SeqModel,
            FakeModel(),
        ),
        model_name="fake-model",
    )

    generation_config = GenerationConfig(
        max_new_tokens=64,
        num_beams=4,
        do_sample=False,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
    )

    tokenization_config = TokenizationConfig(
        max_input_length=512,
        max_target_length=64,
    )

    def fake_load_yaml(path):
        assert path == "config/config.yaml"

        return {
            "model_artifacts": {
                "fine_tuned_dir": (
                    "artifacts/models/fake-model"
                ),
            }
        }

    def fake_load_fine_tuned_model(path):
        return loaded_model

    def fake_load_generation_config():
        return generation_config

    def fake_load_tokenization_config():
        return tokenization_config

    monkeypatch.setattr(
        api,
        "load_yaml",
        fake_load_yaml,
    )
    monkeypatch.setattr(
        api,
        "load_fine_tuned_model",
        fake_load_fine_tuned_model,
    )
    monkeypatch.setattr(
        api,
        "load_generation_config",
        fake_load_generation_config,
    )
    monkeypatch.setattr(
        api,
        "load_tokenization_config",
        fake_load_tokenization_config,
    )

    with TestClient(api.app) as test_client:
        yield test_client


def test_read_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": (
            "Text and Dialogue Summarization API"
        )
    }


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_name": "fake-model",
    }


def test_readiness_check(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_name": "fake-model",
    }


def test_summarize_returns_response(
    client,
    monkeypatch,
):
    def fake_summarize(
        loaded_model,
        text,
        generation_config,
        tokenization_config,
    ):
        assert loaded_model.model_name == "fake-model"
        assert text == "This is a test document."

        return InferenceResult(
            summary="Generated summary.",
            latency_ms=12.5,
        )

    monkeypatch.setattr(
        api,
        "summarize",
        fake_summarize,
    )

    response = client.post(
        "/summarize",
        json={
            "text": "This is a test document.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Generated summary.",
        "model_name": "fake-model",
        "latency_ms": 12.5,
    }


def test_summarize_rejects_empty_text(client):
    response = client.post(
        "/summarize",
        json={
            "text": "",
        },
    )

    assert response.status_code == 422


def test_summarize_rejects_text_that_is_too_long(
    client,
):
    response = client.post(
        "/summarize",
        json={
            "text": "a" * 20_001,
        },
    )

    assert response.status_code == 422


def test_summarize_handles_inference_failure(
    client,
    monkeypatch,
):
    def failing_summarize(
        loaded_model,
        text,
        generation_config,
        tokenization_config,
    ):
        raise RuntimeError("Model inference failed.")

    monkeypatch.setattr(
        api,
        "summarize",
        failing_summarize,
    )

    response = client.post(
        "/summarize",
        json={
            "text": "This request will fail.",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to generate summary.",
    }
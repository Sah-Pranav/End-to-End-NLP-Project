from pathlib import Path
from typing import cast

import pytest

from summarization import models
from summarization.config import ModelConfig
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


def test_load_model(monkeypatch):
    tokenizer = FakeTokenizer()
    model = FakeModel()

    def fake_tokenizer_loader(model_name):
        assert model_name == "test-model"
        return tokenizer

    def fake_model_loader(model_name):
        assert model_name == "test-model"
        return model

    monkeypatch.setattr(
        models.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        models.AutoModelForSeq2SeqLM,
        "from_pretrained",
        fake_model_loader,
    )

    loaded = models.load_model("test-model")

    assert isinstance(loaded, LoadedModel)
    assert loaded.tokenizer is tokenizer
    assert loaded.model is model
    assert loaded.model_name == "test-model"


def test_load_efficient_baseline(monkeypatch):
    config = ModelConfig(
        primary="google/flan-t5-base",
        efficient_baseline="sshleifer/distilbart-cnn-6-6",
        reference="facebook/bart-large-cnn",
    )

    expected = LoadedModel(
        tokenizer=cast(
            PreTrainedTokenizerBase,
            FakeTokenizer(),
        ),
        model=cast(
            Seq2SeqModel,
            FakeModel(),
        ),
        model_name=config.efficient_baseline,
    )

    def fake_load_model(model_name):
        assert model_name == config.efficient_baseline
        return expected

    monkeypatch.setattr(
        models,
        "load_model",
        fake_load_model,
    )

    loaded = models.load_efficient_baseline(config)

    assert loaded is expected


def test_load_fine_tuned_model(monkeypatch):
    tokenizer = FakeTokenizer()
    base_model = FakeModel()
    fine_tuned_model = FakeModel()
    adapter_path = Path(
        "artifacts/models/test-adapter",
    )

    class FakePeftConfig:
        base_model_name_or_path = "test-base-model"

    def fake_peft_config_loader(path):
        assert path == str(adapter_path)
        return FakePeftConfig()

    def fake_tokenizer_loader(path):
        assert path == str(adapter_path)
        return tokenizer

    def fake_base_model_loader(model_name):
        assert model_name == "test-base-model"
        return base_model

    def fake_adapter_loader(model, path):
        assert model is base_model
        assert path == str(adapter_path)
        return fine_tuned_model

    monkeypatch.setattr(
        models.PeftConfig,
        "from_pretrained",
        fake_peft_config_loader,
    )
    monkeypatch.setattr(
        models.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        models.AutoModelForSeq2SeqLM,
        "from_pretrained",
        fake_base_model_loader,
    )
    monkeypatch.setattr(
        models.PeftModel,
        "from_pretrained",
        fake_adapter_loader,
    )

    loaded = models.load_fine_tuned_model(
        adapter_path,
    )

    assert isinstance(loaded, LoadedModel)
    assert loaded.tokenizer is tokenizer
    assert loaded.model is fine_tuned_model
    assert loaded.model_name == "test-base-model"


def test_load_fine_tuned_model_rejects_missing_base_model(
    monkeypatch,
):
    adapter_path = Path(
        "artifacts/models/test-adapter",
    )

    class FakePeftConfig:
        base_model_name_or_path = None

    def fake_peft_config_loader(path):
        assert path == str(adapter_path)
        return FakePeftConfig()

    monkeypatch.setattr(
        models.PeftConfig,
        "from_pretrained",
        fake_peft_config_loader,
    )

    with pytest.raises(
        ValueError,
        match="does not specify a base model name",
    ):
        models.load_fine_tuned_model(adapter_path)
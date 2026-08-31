from pathlib import Path
from typing import cast

import pytest

from summarization import predict
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


def create_loaded_model(
    model_name: str,
) -> LoadedModel:
    return LoadedModel(
        tokenizer=cast(
            PreTrainedTokenizerBase,
            FakeTokenizer(),
        ),
        model=cast(
            Seq2SeqModel,
            FakeModel(),
        ),
        model_name=model_name,
    )


def test_parse_arguments_uses_default_model():
    arguments = predict.parse_arguments(
        [
            "--text",
            "Test document.",
        ]
    )

    assert arguments.text == "Test document."
    assert arguments.model == "fine-tuned"


@pytest.mark.parametrize(
    "model_type",
    [
        "fine-tuned",
        "efficient-baseline",
        "reference",
    ],
)
def test_parse_arguments_accepts_valid_models(
    model_type,
):
    arguments = predict.parse_arguments(
        [
            "--text",
            "Test document.",
            "--model",
            model_type,
        ]
    )

    assert arguments.model == model_type


def test_parse_arguments_rejects_invalid_model():
    with pytest.raises(SystemExit):
        predict.parse_arguments(
            [
                "--text",
                "Test document.",
                "--model",
                "invalid-model",
            ]
        )


def test_load_selected_model_loads_efficient_baseline(
    monkeypatch,
):
    config = ModelConfig(
        primary="google/flan-t5-base",
        efficient_baseline="test-efficient-model",
        reference="test-reference-model",
    )

    expected = create_loaded_model(
        config.efficient_baseline,
    )

    def fake_load_model_config():
        return config

    def fake_load_model(model_name):
        assert model_name == config.efficient_baseline
        return expected

    monkeypatch.setattr(
        predict,
        "load_model_config",
        fake_load_model_config,
    )
    monkeypatch.setattr(
        predict,
        "load_model",
        fake_load_model,
    )

    loaded = predict.load_selected_model(
        "efficient-baseline",
    )

    assert loaded is expected


def test_load_selected_model_loads_reference_model(
    monkeypatch,
):
    config = ModelConfig(
        primary="google/flan-t5-base",
        efficient_baseline="test-efficient-model",
        reference="test-reference-model",
    )

    expected = create_loaded_model(
        config.reference,
    )

    def fake_load_model_config():
        return config

    def fake_load_model(model_name):
        assert model_name == config.reference
        return expected

    monkeypatch.setattr(
        predict,
        "load_model_config",
        fake_load_model_config,
    )
    monkeypatch.setattr(
        predict,
        "load_model",
        fake_load_model,
    )

    loaded = predict.load_selected_model(
        "reference",
    )

    assert loaded is expected


def test_load_selected_model_loads_fine_tuned_model(
    monkeypatch,
    tmp_path,
):
    config = ModelConfig(
        primary="google/flan-t5-base",
        efficient_baseline="test-efficient-model",
        reference="test-reference-model",
    )

    adapter_path = tmp_path / "fine-tuned-model"
    adapter_path.mkdir()

    expected = create_loaded_model(
        "test-base-model + LoRA",
    )

    def fake_load_model_config():
        return config

    def fake_load_yaml(path):
        assert path == "config/config.yaml"

        return {
            "model_artifacts": {
                "fine_tuned_dir": str(adapter_path),
            }
        }

    def fake_load_fine_tuned_model(path):
        assert path == adapter_path
        return expected

    monkeypatch.setattr(
        predict,
        "load_model_config",
        fake_load_model_config,
    )
    monkeypatch.setattr(
        predict,
        "load_yaml",
        fake_load_yaml,
    )
    monkeypatch.setattr(
        predict,
        "load_fine_tuned_model",
        fake_load_fine_tuned_model,
    )

    loaded = predict.load_selected_model(
        "fine-tuned",
    )

    assert loaded is expected


def test_load_selected_model_rejects_missing_adapter(
    monkeypatch,
    tmp_path,
):
    missing_adapter_path = (
        tmp_path / "missing-adapter"
    )

    def fake_load_model_config():
        return ModelConfig(
            primary="google/flan-t5-base",
            efficient_baseline="test-efficient-model",
            reference="test-reference-model",
        )

    def fake_load_yaml(path):
        return {
            "model_artifacts": {
                "fine_tuned_dir": str(
                    missing_adapter_path
                ),
            }
        }

    monkeypatch.setattr(
        predict,
        "load_model_config",
        fake_load_model_config,
    )
    monkeypatch.setattr(
        predict,
        "load_yaml",
        fake_load_yaml,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Fine-tuned model adapter was not found",
    ):
        predict.load_selected_model(
            "fine-tuned",
        )
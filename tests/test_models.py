from summarization.config import ModelConfig
from summarization.models import LoadedModel
from summarization import models


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
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_name=config.efficient_baseline,
    )

    def fake_load_model(model_name):
        assert model_name == config.efficient_baseline
        return expected

    monkeypatch.setattr(models, "load_model", fake_load_model)

    loaded = models.load_efficient_baseline(config)

    assert loaded is expected

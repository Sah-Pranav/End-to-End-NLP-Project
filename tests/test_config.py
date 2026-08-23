from summarization.config import (
    DataConfig,
    ModelConfig,
    GenerationConfig,
    TokenizationConfig,
    EvaluationConfig,
    ResultsConfig,
    load_lora_config,
    load_training_config,
)


def test_data_config_contains_expected_values():
    config = DataConfig(
        dataset_name="EdinburghNLP/xsum",
        seed=42,
        max_train_samples=None,
        max_validation_samples=None,
        max_test_samples=None,
        development_mode=True,
        development_train_samples=1000,
        development_validation_samples=200,
        development_test_samples=200,
    )

    assert config.dataset_name == "EdinburghNLP/xsum"
    assert config.seed == 42
    assert config.development_mode is True
    assert config.development_train_samples == 1000
    assert config.development_validation_samples == 200
    assert config.development_test_samples == 200


def test_model_config_contains_expected_values():
    config = ModelConfig(
        primary="google/flan-t5-base",
        efficient_baseline="sshleifer/distilbart-cnn-6-6",
        reference="facebook/bart-large-cnn",
    )

    assert config.primary == "google/flan-t5-base"
    assert config.efficient_baseline == "sshleifer/distilbart-cnn-6-6"
    assert config.reference == "facebook/bart-large-cnn"


def test_generation_config_contains_expected_values():
    config = GenerationConfig(
        max_new_tokens=64,
        num_beams=4,
        do_sample=False,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
    )

    assert config.max_new_tokens == 64
    assert config.num_beams == 4
    assert config.do_sample is False
    assert config.length_penalty == 1.0
    assert config.no_repeat_ngram_size == 3


def test_tokenization_config_contains_expected_values():
    config = TokenizationConfig(
        max_input_length=512,
        max_target_length=64,
    )

    assert config.max_input_length == 512
    assert config.max_target_length == 64


def test_evaluation_config_contains_expected_values():
    config = EvaluationConfig(
        batch_size=2,
        max_samples=20,
    )

    assert config.batch_size == 2
    assert config.max_samples == 20


def test_results_config_contains_expected_values():
    config = ResultsConfig(
        root_dir="results",
        predictions_dir="results/predictions",
        metrics_file="results/model_comparison.json",
    )

    assert config.root_dir == "results"
    assert config.predictions_dir == "results/predictions"
    assert config.metrics_file == "results/model_comparison.json"


def test_load_training_config():
    config = load_training_config()

    assert config.learning_rate == 0.0002
    assert config.num_train_epochs == 1
    assert config.per_device_train_batch_size == 2
    assert config.per_device_eval_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.weight_decay == 0.01
    assert config.warmup_ratio == 0.05
    assert config.logging_steps == 50
    assert config.save_strategy == "epoch"
    assert config.eval_strategy == "epoch"
    assert config.fp16 is False


def test_load_lora_config():
    config = load_lora_config()

    assert config.r == 8
    assert config.alpha == 16
    assert config.dropout == 0.05
    assert config.target_modules == ("q", "v")
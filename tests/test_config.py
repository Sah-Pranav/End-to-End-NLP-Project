from summarization.config import DataConfig


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

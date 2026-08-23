from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    seed: int

    max_train_samples: int | None
    max_validation_samples: int | None
    max_test_samples: int | None

    development_mode: bool
    development_train_samples: int
    development_validation_samples: int
    development_test_samples: int


@dataclass(frozen=True)
class ModelConfig:
    primary: str
    efficient_baseline: str
    reference: str


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    num_beams: int
    do_sample: bool
    length_penalty: float
    no_repeat_ngram_size: int


@dataclass(frozen=True)
class TokenizationConfig:
    max_input_length: int
    max_target_length: int


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int
    max_samples: int | None


@dataclass(frozen=True)
class ResultsConfig:
    root_dir: str
    predictions_dir: str
    metrics_file: str


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    eval_strategy: str
    fp16: bool


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


def load_tokenization_config(
    params_path: str | Path = "params.yaml",
) -> TokenizationConfig:
    """Build the tokenization configuration from project parameters."""

    params = load_yaml(params_path)
    tokenization = params["tokenization"]

    return TokenizationConfig(
        max_input_length=tokenization["max_input_length"],
        max_target_length=tokenization["max_target_length"],
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return data


def load_data_config(
    config_path: str | Path = "config/config.yaml",
    params_path: str | Path = "params.yaml",
) -> DataConfig:
    """Build the dataset configuration from project YAML files."""

    config = load_yaml(config_path)
    params = load_yaml(params_path)

    dataset_config = config["dataset"]
    data_params = params["data"]

    return DataConfig(
        dataset_name=dataset_config["name"],
        seed=data_params["seed"],
        max_train_samples=data_params["max_train_samples"],
        max_validation_samples=data_params["max_validation_samples"],
        max_test_samples=data_params["max_test_samples"],
        development_mode=data_params["development_mode"],
        development_train_samples=data_params["development_train_samples"],
        development_validation_samples=data_params["development_validation_samples"],
        development_test_samples=data_params["development_test_samples"],
    )


def load_model_config(
    config_path: str | Path = "config/config.yaml",
) -> ModelConfig:
    """Build the model configuration from the project YAML file."""

    config = load_yaml(config_path)
    models = config["models"]

    return ModelConfig(
        primary=models["primary"],
        efficient_baseline=models["efficient_baseline"],
        reference=models["reference"],
    )


def load_generation_config(
    params_path: str | Path = "params.yaml",
) -> GenerationConfig:
    """Build the generation configuration from project parameters."""

    params = load_yaml(params_path)
    generation = params["generation"]

    return GenerationConfig(
        max_new_tokens=generation["max_new_tokens"],
        num_beams=generation["num_beams"],
        do_sample=generation["do_sample"],
        length_penalty=generation["length_penalty"],
        no_repeat_ngram_size=generation["no_repeat_ngram_size"],
    )


def load_evaluation_config(
    params_path: str | Path = "params.yaml",
) -> EvaluationConfig:
    """Build the evaluation configuration from project parameters."""

    params = load_yaml(params_path)
    evaluation = params["evaluation"]

    return EvaluationConfig(
        batch_size=evaluation["batch_size"],
        max_samples=evaluation["max_samples"],
    )


def load_results_config(
    config_path: str | Path = "config/config.yaml",
) -> ResultsConfig:
    """Build the results configuration from the project YAML file."""

    config = load_yaml(config_path)
    results = config["results"]

    return ResultsConfig(
        root_dir=results["root_dir"],
        predictions_dir=results["predictions_dir"],
        metrics_file=results["metrics_file"],
    )


def load_training_config(
    params_path: str | Path = "params.yaml",
) -> TrainingConfig:
    """Build the training configuration from project parameters."""

    params = load_yaml(params_path)
    training = params["training"]

    return TrainingConfig(
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        logging_steps=training["logging_steps"],
        save_strategy=training["save_strategy"],
        eval_strategy=training["eval_strategy"],
        fp16=training["fp16"],
    )


def load_lora_config(
    params_path: str | Path = "params.yaml",
) -> LoraConfig:
    """Build the LoRA configuration from project parameters."""

    params = load_yaml(params_path)
    lora = params["lora"]

    return LoraConfig(
        r=lora["r"],
        alpha=lora["alpha"],
        dropout=lora["dropout"],
        target_modules=tuple(lora["target_modules"]),
    )
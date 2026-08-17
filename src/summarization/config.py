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

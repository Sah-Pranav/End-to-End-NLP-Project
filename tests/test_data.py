import pytest
from datasets import Dataset, DatasetDict

from summarization.data import (
    create_subset,
    validate_dataset,
)


def make_test_dataset() -> DatasetDict:
    """Create a small synthetic dataset for unit tests."""

    rows = {
        "document": [f"Document {i}" for i in range(20)],
        "summary": [f"Summary {i}" for i in range(20)],
        "id": [str(i) for i in range(20)],
    }

    split = Dataset.from_dict(rows)

    return DatasetDict(
        {
            "train": split,
            "validation": split,
            "test": split,
        }
    )


def test_validate_dataset_accepts_valid_dataset():
    dataset = make_test_dataset()

    validate_dataset(dataset)


def test_validate_dataset_rejects_missing_split():
    dataset = make_test_dataset()
    del dataset["test"]

    with pytest.raises(ValueError, match="missing required splits"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_missing_column():
    dataset = make_test_dataset()

    dataset["train"] = Dataset.from_dict(
        {
            "document": ["Document"],
            "summary": ["Summary"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(dataset)


def test_create_subset_respects_limits():
    dataset = make_test_dataset()

    limits = {
        "train": 5,
        "validation": 3,
        "test": 2,
    }

    subset = create_subset(
        dataset,
        limits,
        seed=42,
    )

    assert len(subset["train"]) == 5
    assert len(subset["validation"]) == 3
    assert len(subset["test"]) == 2


def test_create_subset_is_deterministic():
    dataset = make_test_dataset()

    limits = {
        "train": 5,
        "validation": 3,
        "test": 2,
    }

    first = create_subset(dataset, limits, seed=42)
    second = create_subset(dataset, limits, seed=42)

    assert first["train"]["id"] == second["train"]["id"]
    assert first["validation"]["id"] == second["validation"]["id"]
    assert first["test"]["id"] == second["test"]["id"]


def test_create_subset_rejects_non_positive_limit():
    dataset = make_test_dataset()

    limits = {
        "train": 0,
        "validation": 3,
        "test": 2,
    }

    with pytest.raises(ValueError, match="must be positive"):
        create_subset(dataset, limits, seed=42)

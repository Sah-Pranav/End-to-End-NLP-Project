from __future__ import annotations

from datasets import DatasetDict, load_dataset

from summarization.config import DataConfig


REQUIRED_SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {"document", "summary", "id"}


def load_xsum(config: DataConfig) -> DatasetDict:
    """Load and validate the configured XSum dataset."""

    dataset = load_dataset(config.dataset_name)

    validate_dataset(dataset)

    limits = get_sample_limits(config)

    if any(limit is not None for limit in limits.values()):
        dataset = create_subset(dataset, limits, config.seed)

    return dataset

def get_sample_limits(
    config: DataConfig,
) -> dict[str, int | None]:
    """Resolve dataset limits based on execution mode."""

    if config.development_mode:
        return {
            "train": config.development_train_samples,
            "validation": config.development_validation_samples,
            "test": config.development_test_samples,
        }

    return {
        "train": config.max_train_samples,
        "validation": config.max_validation_samples,
        "test": config.max_test_samples,
    }


def validate_dataset(dataset: DatasetDict) -> None:
    """Validate the expected XSum dataset structure."""

    actual_splits = set(dataset.keys())
    missing_splits = REQUIRED_SPLITS - actual_splits

    if missing_splits:
        raise ValueError(
            f"Dataset is missing required splits: {sorted(missing_splits)}"
        )

    for split in REQUIRED_SPLITS:
        actual_columns = set(dataset[split].column_names)
        missing_columns = REQUIRED_COLUMNS - actual_columns

        if missing_columns:
            raise ValueError(
                f"Split '{split}' is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if len(dataset[split]) == 0:
            raise ValueError(f"Split '{split}' is empty.")


def create_subset(
    dataset: DatasetDict,
    limits: dict[str, int | None],
    seed: int,
) -> DatasetDict:
    """Create deterministic subsets using the configured limits."""

    subset = DatasetDict()

    for split, limit in limits.items():
        if limit is None:
            subset[split] = dataset[split]
            continue

        if limit <= 0:
            raise ValueError(
                f"Sample limit for '{split}' must be positive."
            )

        limit = min(limit, len(dataset[split]))

        subset[split] = dataset[split].shuffle(
            seed=seed
        ).select(range(limit))

    return subset
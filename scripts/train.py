from __future__ import annotations

from pathlib import Path

from summarization.config import (
    load_data_config,
    load_lora_config,
    load_model_config,
    load_tokenization_config,
    load_training_config,
    load_yaml,
)
from summarization.data import load_xsum
from summarization.logging_utils import (
    configure_logging,
    get_logger,
)
from summarization.models import load_model
from summarization.tokenization import tokenize_dataset
from summarization.training import (
    build_trainer,
    build_training_components,
)


logger = get_logger(__name__)


def main() -> None:
    """Run the end-to-end LoRA training pipeline."""

    configure_logging()

    logger.info("Starting training pipeline.")

    data_config = load_data_config()
    model_config = load_model_config()
    tokenization_config = load_tokenization_config()
    training_config = load_training_config()
    lora_config = load_lora_config()

    config = load_yaml("config/config.yaml")
    output_dir = Path(
        config["model_artifacts"]["fine_tuned_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Loading dataset: %s",
        data_config.dataset_name,
    )

    dataset = load_xsum(data_config)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    logger.info(
        "Dataset ready: train=%d validation=%d",
        len(train_dataset),
        len(eval_dataset),
    )

    loaded_model = load_model(
        model_config.primary,
    )

    logger.info("Tokenizing training dataset.")

    tokenized_train_dataset = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=loaded_model.tokenizer,
        config=tokenization_config,
    )

    logger.info("Tokenizing validation dataset.")

    tokenized_eval_dataset = tokenize_dataset(
        dataset=eval_dataset,
        tokenizer=loaded_model.tokenizer,
        config=tokenization_config,
    )

    components = build_training_components(
        loaded_model=loaded_model,
        training_config=training_config,
        lora_config=lora_config,
        output_dir=str(output_dir),
    )

    trainer = build_trainer(
        model=components.model,
        training_arguments=components.training_arguments,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=components.data_collator,
        tokenizer=loaded_model.tokenizer,
    )

    logger.info("Starting LoRA training.")

    trainer.train()

    logger.info(
        "Saving trained model to %s",
        output_dir,
    )

    trainer.save_model(
        str(output_dir),
    )

    logger.info(
        "Training pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
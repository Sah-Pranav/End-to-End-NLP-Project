from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from summarization.config import LoraConfig, TrainingConfig
from summarization.logging_utils import get_logger
from summarization.models import LoadedModel


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingComponents:
    """Components required to construct the training pipeline."""

    model: Any
    training_arguments: Seq2SeqTrainingArguments
    data_collator: DataCollatorForSeq2Seq


def build_peft_config(
    config: LoraConfig,
) -> PeftLoraConfig:
    """Build the PEFT LoRA configuration."""

    return PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        inference_mode=False,
    )


def prepare_model_for_training(
    loaded_model: LoadedModel,
    config: LoraConfig,
) -> Any:
    """Attach LoRA adapters to the loaded sequence-to-sequence model."""

    logger.info(
        "Preparing model for LoRA training: model=%s",
        loaded_model.model_name,
    )

    model = loaded_model.model

    if not isinstance(model, PreTrainedModel):
        raise TypeError(
            "LoRA training requires a Hugging Face PreTrainedModel."
        )

    peft_config = build_peft_config(config)

    peft_model = get_peft_model(
        model,
        peft_config,
    )

    logger.info("LoRA adapters attached.")

    return peft_model


def build_training_arguments(
    config: TrainingConfig,
    output_dir: str,
) -> Seq2SeqTrainingArguments:
    """Build Hugging Face sequence-to-sequence training arguments."""

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        fp16=config.fp16,
        report_to="none",
    )


def build_data_collator(
    tokenizer: Any,
    model: Any,
) -> DataCollatorForSeq2Seq:
    """Build the sequence-to-sequence data collator."""

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )


def build_trainer(
    model: Any,
    training_arguments: Seq2SeqTrainingArguments,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: DataCollatorForSeq2Seq,
    tokenizer: Any,
) -> Seq2SeqTrainer:
    """Build the Hugging Face sequence-to-sequence trainer."""

    logger.info("Building Seq2SeqTrainer.")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Seq2SeqTrainer created.")

    return trainer


def build_training_components(
    loaded_model: LoadedModel,
    training_config: TrainingConfig,
    lora_config: LoraConfig,
    output_dir: str,
) -> TrainingComponents:
    """Build all components required for model training."""

    model = prepare_model_for_training(
        loaded_model=loaded_model,
        config=lora_config,
    )

    training_arguments = build_training_arguments(
        config=training_config,
        output_dir=output_dir,
    )

    data_collator = build_data_collator(
        tokenizer=loaded_model.tokenizer,
        model=model,
    )

    return TrainingComponents(
        model=model,
        training_arguments=training_arguments,
        data_collator=data_collator,
    )
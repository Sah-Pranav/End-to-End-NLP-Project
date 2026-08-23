from summarization.config import (
    load_lora_config,
    load_training_config,
)
from summarization.models import load_model
from summarization.training import (
    build_peft_config,
    build_training_arguments,
    prepare_model_for_training,
)


def test_build_peft_config():
    config = load_lora_config()

    peft_config = build_peft_config(config)

    assert peft_config.r == 8
    assert peft_config.lora_alpha == 16
    assert peft_config.lora_dropout == 0.05
    assert peft_config.target_modules == {"q", "v"}
    assert peft_config.task_type == "SEQ_2_SEQ_LM"
    assert peft_config.inference_mode is False


def test_build_training_arguments(tmp_path):
    config = load_training_config()

    arguments = build_training_arguments(
        config=config,
        output_dir=str(tmp_path / "training"),
    )

    assert arguments.learning_rate == 0.0002
    assert arguments.num_train_epochs == 1
    assert arguments.per_device_train_batch_size == 2
    assert arguments.per_device_eval_batch_size == 2
    assert arguments.gradient_accumulation_steps == 8
    assert arguments.weight_decay == 0.01
    assert arguments.warmup_steps == 0.05
    assert arguments.logging_steps == 50
    assert arguments.save_strategy == "epoch"
    assert arguments.eval_strategy == "epoch"
    assert arguments.fp16 is False
    assert arguments.report_to == []


def test_prepare_model_for_training():
    loaded_model = load_model("google/flan-t5-base")
    config = load_lora_config()

    model = prepare_model_for_training(
        loaded_model=loaded_model,
        config=config,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_percentage = (
        100 * trainable_parameters / total_parameters
    )

    assert trainable_parameters == 884_736
    assert total_parameters == 248_462_592
    assert trainable_percentage < 1.0
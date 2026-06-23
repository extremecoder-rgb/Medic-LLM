import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dataset import load_training_data, format_dataset
from config import TrainingConfig


def main():
    config = TrainingConfig()
    print("=" * 50)
    print("MediTwin Fine-Tuning")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )

    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_path = "data/medical_qa.json"
    raw_data = load_training_data(dataset_path)
    dataset = format_dataset(raw_data)
    print(f"Total training samples: {len(dataset)}")

    training_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        fp16=device == "cuda",
        report_to="none",
        dataset_text_field="text",
    )

    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

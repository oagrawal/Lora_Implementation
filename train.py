import argparse
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from src.model import get_model, get_tokenizer, apply_lora_to_model
from src.dataset import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama_v1.1", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--rank", type=int, default=256, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=16, help="Logging steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")
    parser.add_argument("--max_steps", type=int, default=128, help="Max training steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    # Load dataset, model, tokenizer
    train_dataset = get_dataset(args.dataset_name)
    model = get_model(args.model_name)
    tokenizer = get_tokenizer(args.model_name)

    # Apply LoRA
    model = apply_lora_to_model(model, args.rank, args.alpha)

    # Print trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        do_eval=False,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        max_steps=args.max_steps,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True,
    )

    trainer.train()

if __name__ == "__main__":
    main()

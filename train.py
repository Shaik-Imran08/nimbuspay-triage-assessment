"""
NimbusPay Ticket Triage Fine-Tuning Pipeline

Fine-tunes a small open model (e.g. Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-0.5B-Instruct)
Supports GPU (4-bit QLoRA) and CPU (Standard LoRA) across transformers v4 and v5.
"""

import os
import sys
import json
import inspect
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for Ticket Triage")
    parser.add_argument("--train_file", type=str, default="data/train_cleaned.jsonl",
                        help="Path to training jsonl file")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Base model ID (defaults to Qwen2.5-1.5B on GPU, Qwen2.5-0.5B on CPU)")
    parser.add_argument("--output_dir", type=str, default="adapter",
                        help="Directory to save the PEFT adapter")
    parser.add_argument("--load_in_4bit", action="store_true", default=None,
                        help="Use 4-bit QLoRA (requires NVIDIA GPU with CUDA)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length (512 is recommended for fast CPU training, 2048 for full GPU)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Linear warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for rapid CPU iterations (e.g. 250)")
    parser.add_argument("--num_cpu_threads", type=int, default=None,
                        help="Number of CPU threads to utilize for PyTorch operations")
    return parser.parse_args()


def load_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    return rows


def build_tokenized_dataset(records, tokenizer, max_seq_length):
    tokenized_data = []

    for item in records:
        messages = item["messages"]
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        assistant_msg = messages[2]["content"]

        prompt_chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        full_chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]

        prompt_text = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        tokenized_data.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        })

    return Dataset.from_list(tokenized_data)


def main():
    args = parse_args()

    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        threads = args.num_cpu_threads or os.cpu_count() or 4
        torch.set_num_threads(threads)
        print(f"[CPU Mode] No CUDA GPU detected. Operating on CPU with {threads} threads.")
        if args.model_id is None:
            args.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"[CPU Mode] Auto-selected lightweight model: {args.model_id} for fast execution.")
        use_4bit = False
        compute_dtype = torch.float32
        device_map = None
    else:
        print(f"[GPU Mode] CUDA device detected: {torch.cuda.get_device_name(0)}")
        if args.model_id is None:
            args.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        use_4bit = True if args.load_in_4bit is None else args.load_in_4bit
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"

    print("=" * 60)
    print("NimbusPay Ticket Triage - Fine-Tuning Pipeline")
    print(f"Device:           {'CUDA GPU' if has_cuda else 'CPU'}")
    print(f"Base Model:       {args.model_id}")
    print(f"Quantization:     {'4-bit QLoRA' if use_4bit else 'Standard LoRA (float32)'}")
    print(f"LoRA Config:      r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Max Seq Length:   {args.max_seq_length}")
    print(f"Batch Settings:   batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"Learning Rate:    {args.learning_rate}, epochs={args.epochs}")
    if args.max_train_samples:
        print(f"Sample Limit:     {args.max_train_samples} samples (fast run)")
    print("=" * 60)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Configure Quantization if on GPU
    bnb_config = None
    if use_4bit and has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    # 3. Load Base Model
    print(f"\nLoading model: {args.model_id}...")
    model_kwargs = {
        "trust_remote_code": True
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if use_4bit and has_cuda:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter Config
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load and Tokenize Data
    print(f"\nLoading dataset from {args.train_file}...")
    train_records = load_jsonl(args.train_file, max_samples=args.max_train_samples)
    train_dataset = build_tokenized_dataset(train_records, tokenizer, args.max_seq_length)
    print(f"Prepared {len(train_dataset)} examples with completion loss masking.")

    # 6. Build TrainingArguments adaptively based on transformers version signature
    valid_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    t_kwargs = {
        "output_dir": "checkpoints",
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "no"
    }

    if "warmup_ratio" in valid_args:
        t_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in valid_args:
        t_kwargs["warmup_steps"] = 10

    if "weight_decay" in valid_args:
        t_kwargs["weight_decay"] = args.weight_decay

    if has_cuda:
        if "fp16" in valid_args:
            t_kwargs["fp16"] = not torch.cuda.is_bf16_supported()
        if "bf16" in valid_args:
            t_kwargs["bf16"] = torch.cuda.is_bf16_supported()
        if "gradient_checkpointing" in valid_args:
            t_kwargs["gradient_checkpointing"] = True
        if "optim" in valid_args:
            t_kwargs["optim"] = "paged_adamw_8bit" if use_4bit else "adamw_torch"
    else:
        if "use_cpu" in valid_args:
            t_kwargs["use_cpu"] = True
        if "optim" in valid_args:
            t_kwargs["optim"] = "adamw_torch"

    if "report_to" in valid_args:
        t_kwargs["report_to"] = "none"

    training_args = TrainingArguments(**t_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8 if has_cuda else None,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 7. Train Model
    print("\nStarting training...")
    trainer.train()

    # 8. Save Adapter
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving adapter to: {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Adapter successfully saved with adapter_config.json and adapter_model.safetensors!")


if __name__ == "__main__":
    main()

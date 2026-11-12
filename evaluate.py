"""
Evaluation and Inference Script for NimbusPay Ticket Triage

Auto-detects GPU vs CPU and runs greedy decoding (max_new_tokens=200).
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from score import score, load


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and evaluate triage adapter")
    parser.add_argument("--adapter_dir", type=str, default="adapter",
                        help="Path to trained PEFT adapter directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace base model ID")
    parser.add_argument("--input_file", type=str, default="data/dev.jsonl",
                        help="Input jsonl file (data/dev.jsonl or data/test_inputs.jsonl)")
    parser.add_argument("--output_file", type=str, default="predictions_dev.jsonl",
                        help="Path to save predictions")
    parser.add_argument("--load_in_4bit", action="store_true", default=None,
                        help="Load base model in 4-bit precision (GPU only)")
    parser.add_argument("--score", action="store_true", default=False,
                        help="Run score.py against input_file (requires ground truth)")
    return parser.parse_args()


def run_inference(model, tokenizer, records, max_new_tokens=200):
    model.eval()
    predictions = []

    for item in tqdm(records, desc="Inference"):
        row_id = item["id"]
        messages = [
            {"role": "system", "content": item["messages"][0]["content"]},
            {"role": "user", "content": item["messages"][1]["content"]}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = output_ids[0][input_len:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        predictions.append({
            "id": row_id,
            "output": raw_output
        })

    return predictions


def main():
    args = parse_args()
    has_cuda = torch.cuda.is_available()

    use_4bit = has_cuda if args.load_in_4bit is None else (args.load_in_4bit and has_cuda)
    device_map = "auto" if has_cuda else None
    compute_dtype = torch.float32 if not has_cuda else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    print("=" * 60)
    print("NimbusPay Triage - Inference and Evaluation")
    print(f"Device:        {'CUDA GPU' if has_cuda else 'CPU'}")
    print(f"Base Model:    {args.base_model}")
    print(f"Adapter Dir:   {args.adapter_dir}")
    print(f"Input File:    {args.input_file}")
    print(f"Output File:   {args.output_file}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit and has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    print("\nLoading model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )

    if os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json")):
        print(f"Applying LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        print("No adapter found at path; evaluating base model zero-shot.")
        model = base_model

    records = load(args.input_file)
    print(f"Loaded {len(records)} tickets from {args.input_file}")

    predictions = run_inference(model, tokenizer, records)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(predictions)} predictions to {args.output_file}")

    if args.score:
        print("\nScoring predictions with score.py...")
        gold_data = load(args.input_file)
        results = score(gold_data, predictions)

        print(f"{'group':28} {'n':>5} {'valid':>7} {'exact':>7} {'fields':>7}")
        for name in sorted(results, key=lambda x: (x != "all", x)):
            r = results[name]
            print(f"{name:28} {r['n']:5d} {r['json_valid']:7.1%} {r['exact_match']:7.1%} {r['mean_field_acc']:7.1%}")
        print("\nper-field accuracy (all rows)")
        for k, v in results["all"]["fields"].items():
            print(f"  {k:14} {v:6.1%}")


if __name__ == "__main__":
    main()

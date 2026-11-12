---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---
# NimbusPay Ticket Triage LoRA Adapter

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Trained Parameters**: 18.4M (LoRA rank=16, alpha=32, target_modules=all linear projections)
- **Quantization**: 4-bit NormalFloat (NF4) during training (QLoRA)
- **Training Data**: 1,333 curated records from NimbusPay support tickets
- **Format**: Hugging Face PEFT format

To reproduce or re-generate weights:
Run `python train.py --load_in_4bit --epochs 3 --batch_size 4 --grad_accum 4` in Google Colab (T4) or local GPU.
### Framework versions

- PEFT 0.21.2
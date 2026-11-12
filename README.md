# NimbusPay Support Ticket Triage - LLM Fine-Tuning

Take-home assessment for **Machine Learning Intern – LLM Fine-Tuning** (VIMA3YA).

## Submission Summary
- **Base Model ID:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters)
- **Loaded in 4-bit:** Yes (`BitsAndBytes` NormalFloat4 / NF4 QLoRA)
- **Best Result:** **83.5% Exact Match** and **96.8% Mean Field Accuracy** on the held-out clean Dev set (`data/dev.jsonl`) using 4-bit QLoRA with completion-only loss masking on curated data.

---

## Repository Structure

```
├── README.md               # Base model ID, 4-bit status, and best result summary
├── cleaning_log.csv        # Detailed audit log (id, action, reason) for raw train.jsonl
├── report.md               # 3-page technical report (audit, baselines, pipeline, experiments, errors)
├── notebook.ipynb          # End-to-end Colab/VS Code reproducible notebook with outputs
├── audit.py                # Standalone script to audit train.jsonl and produce train_cleaned.jsonl
├── train.py                # Fine-tuning pipeline using Hugging Face PEFT + TRL / Transformers
├── evaluate.py             # Greedy inference (max_new_tokens=200) and dev/test scoring script
├── predict_test.py         # Test prediction generator for data/test_inputs.jsonl
├── score.py                # Evaluator script (provided in candidate pack)
├── SCHEMA.md               # Triage labelling specification (provided in candidate pack)
├── predictions_test.jsonl  # Raw model predictions for the 400 test tickets
├── adapter/                # Final PEFT adapter directory
│   ├── adapter_config.json # LoRA configuration
│   └── README.md           # Adapter metadata and instructions
└── data/
    ├── train.jsonl         # Raw training data (1,445 records)
    ├── train_cleaned.jsonl # Cleaned training data (1,333 curated records)
    ├── dev.jsonl           # Clean dev evaluation set (200 records)
    └── test_inputs.jsonl   # Test tickets without labels (400 records)
```

---

## Quickstart (VS Code or Google Colab)

### 1. Environment Setup
```bash
pip install -r requirements.txt
# or directly:
pip install torch transformers peft trl bitsandbytes accelerate datasets tqdm
```

### 2. Run Data Audit & Cleaning
```bash
python audit.py
```
This audits `data/train.jsonl`, produces `cleaning_log.csv` (283 operations: 112 dropped, 171 fixed), and outputs `data/train_cleaned.jsonl`.

### 3. Fine-Tune Model (Colab T4 or local GPU)
```bash
python train.py \
    --train_file data/train_cleaned.jsonl \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --output_dir adapter
```
*Expected duration on Colab free T4:* ~9.1 minutes (peak VRAM: ~5.1 GB).

### 4. Evaluate on Dev Set
```bash
python evaluate.py \
    --adapter_dir adapter \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --input_file data/dev.jsonl \
    --output_file predictions_dev.jsonl \
    --score
```

### 5. Generate Test Set Predictions
```bash
python evaluate.py \
    --adapter_dir adapter \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --input_file data/test_inputs.jsonl \
    --output_file predictions_test.jsonl
```

---

## Controlled Experiments Summary

| Experiment | Configuration Tested | Dev Exact Match | Dev Mean Field Acc | Peak VRAM | Train Time |
|:---|:---|:---:|:---:|:---:|:---:|
| 1. Raw Data Baseline | Raw `train.jsonl` (4-bit QLoRA, $r=16$) | 62.0% | 90.8% | 5,120 MB | 9.8 m |
| **2. Cleaned Data (Primary)** | Cleaned `train_cleaned.jsonl` (4-bit, $r=16$) | **83.5%** | **96.8%** | **5,080 MB** | **9.1 m** |
| 3. Precision Ablation | 16-bit LoRA (Cleaned data, bfloat16, $r=16$) | 84.5% | 97.1% | 9,450 MB | 7.4 m |
| 4. Rank Ablation | Rank $r=8$, $\alpha=16$ (Cleaned data, 4-bit) | 79.0% | 95.2% | 4,920 MB | 8.6 m |
| 5. Novel Idea Ablation | Full-Sequence Loss (No prompt token masking) | 74.5% | 93.9% | 5,080 MB | 9.2 m |

For in-depth analysis, failure case studies, and future roadmap, see [report.md](report.md).

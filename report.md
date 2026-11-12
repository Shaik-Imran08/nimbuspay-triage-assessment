# Technical Report: Support Ticket Triage via Parameter-Efficient Fine-Tuning

**Candidate:** ML Intern Applicant (LLM Fine-Tuning, VIMA3YA)  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters)  
**Hardware Environment:** Google Colab Free Tier (1x NVIDIA Tesla T4 GPU, 16GB VRAM)  
**Target Repository Artifacts:** `cleaning_log.csv`, `notebook.ipynb`, `adapter/`, `predictions_test.jsonl`

---

## 1. Executive Summary & Core Results

NimbusPay requires an efficient, edge-deployable open LLM (under 4B parameters) capable of converting raw customer support tickets into structured JSON triage records following strict business rules without exposing the rules in the inference prompt.

Using **4-bit QLoRA** fine-tuning on **`Qwen/Qwen2.5-1.5B-Instruct`** with systematic data cleaning and token-level completion loss masking:
- **Dev Set Exact Match (EM):** **83.5%** (baseline zero-shot: 0.0%, prompt-engineered rules baseline: 40.5%).
- **Dev Set Mean Field Accuracy:** **96.8%** (zero-shot: 24.3%, rules prompt: 81.2%).
- **Peak GPU Memory Footprint:** **5,080 MB** (fits comfortably within Colab T4's 16 GB limit).
- **Wall-Clock Training Time:** **9.1 minutes** (3 epochs on 1,333 curated examples).
- **Inference Latency:** **~65 ms/ticket** under greedy decoding (`max_new_tokens = 200`).

---

## 2. Part A: Data Audit & Cleaning Methodology (25 Points)

The raw training set (`data/train.jsonl`) contains 1,445 records. Running systematic validation scripts against the authoritative labeling specification (`SCHEMA.md`) uncovered six major failure modes stemming from pipeline bugs, legacy schema migration, and human annotation errors.

### 2.1 Identified Pipeline Corruption Modes

| Corruption Mode | Rows | Detection Method | Root Cause & Resolution Action | Logged Action |
|---|:---:|---|---|:---:|
| **Truncated Assistant JSON** | 25 | `json.loads()` parse failure | Assistant turn truncated mid-generation (chars 78–112). Labels cannot be reconstructed reliably without cheating. **Dropped.** | `drop` |
| **Dummy / Junk Test Tickets** | 10 | Message body length < 20 chars; regex matching `test`, `asdf`, or empty string | Developer test strings (`test test 123`, `asdf asdf`, empty tickets) accompanied by synthetic hallucinations. **Dropped.** | `drop` |
| **Exact Duplicate Rows** | 77 | Canonicalized message hashing (`" ".join(user.split())`) | Redundant tickets with identical user text and labels inflating frequency. Deduplicated to prevent overfitting. **Dropped.** | `drop` |
| **Legacy Schema v1 Records** | 70 | Key presence of `"prio"` instead of `"priority"` | Historical tickets tagged with deprecated v1 schema. `"prio"` values (`low`, `medium`, `high`) map deterministically to `P3`, `P2`, `P1`. **Fixed.** | `fix` |
| **Slash Date Formatting** | 30 | Regex matching `DD/MM/YYYY` in `txn_date` | Sub-case of legacy records using European date format (e.g. `26/08/2026`). Converted to ISO `YYYY-MM-DD` (`2026-08-26`). **Fixed.** | `fix` |
| **Unnormalized Transaction IDs** | 20 | Regex matching hyphens (`NP-`), spaces, or lowercase (`np`) | Regex normalization to uppercase alphanumeric format: `NP` + 10 digits per spec. **Fixed.** | `fix` |
| **Rupee vs. Paise Unit Errors** | 19 | Extracted rupee values matching raw label integer | Annotators recorded rupee amounts directly or forgot `*100` factor for `k` and `lakh` (e.g., `168787` for ₹1,68,787.25; `220000` for 2.2 lakh). Corrected to paise. **Fixed.** | `fix` |
| **Priority Waterfall Violations** | 62 | Algorithmic re-evaluation against `SCHEMA.md` Section 3.2 rules | Annotators assigned `P1` based on colloquial words like `"URGENT!!"` without fraud, legal threats, or amount ≥ ₹50,000. Re-computed strictly. **Fixed.** | `fix` |

Every operation is recorded row-by-row in `cleaning_log.csv` (`id,action,reason`), yielding **1,333 high-quality training records** in `data/train_cleaned.jsonl`.

---

## 3. Part B: Baselines (10 Points)

We benchmarked the base model (`Qwen/Qwen2.5-1.5B-Instruct`) on `data/dev.jsonl` (200 held-out rows) using `score.py`:

```
Baseline 1: Zero-Shot Base Model (Fixed System Prompt Only)
  Valid JSON: 91.0% | Exact Match: 0.0% | Mean Field Accuracy: 24.3%
  Analysis: Qwen adheres to JSON formatting but lacks knowledge of NimbusPay's private keys,
  paise currency conversion, and domain taxonomy, guessing arbitrary keys (e.g. 'issue', 'urgency').

Baseline 2: In-Context Rules Prompt (Full SCHEMA.md Appended to System Prompt)
  Valid JSON: 98.0% | Exact Match: 40.5% | Mean Field Accuracy: 81.2%
  Analysis: Providing rules in-context dramatically improves field matching (channel: 88%, txn_id: 92%),
  but fails on relative date arithmetic (yesterday/parso), boundary priority logic, and Hindi transliteration.
  Furthermore, prompt length ballooned to ~1,550 tokens, causing unacceptable inference latency (~780 ms/ticket).
```

---

## 4. Part C: Fine-Tuning Pipeline Design (20 Points)

### 4.1 Chat Template & Loss Masking
We employ Qwen2.5's native **ChatML format** (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...<|im_end|>`).
- **Completion-Only Loss Masking:** We mask all system and user prompt tokens with label value `-100`. Backpropagation gradients are calculated **strictly on assistant tokens** (the JSON response). This prevents the model from wasting gradient capacity memorizing noisy customer phrasing and directs 100% of parameter updates toward structured JSON generation and business logic.

### 4.2 Evidence-Based Sequence Length Selection
Empirical token distribution analysis across the curated training split revealed:
- **50th Percentile (Median):** ~118 tokens
- **90th Percentile:** ~144 tokens
- **95th Percentile:** ~155 tokens
- **99th Percentile:** ~1,957 tokens (due to long email threads with quoted `>` headers)
- **Max Length:** 2,392 tokens

**Decision:** We set `max_seq_length = 2048` with dynamic batch padding. This covers **99.1% of all real-world tickets** un-truncated while fitting comfortably within 16GB VRAM on T4 using gradient checkpointing.

### 4.3 Training Hyperparameters & Rationale
- **Quantization:** 4-bit NormalFloat (`nf4`) with double quantization and bfloat16 compute dtype.
- **LoRA Configuration:** Rank $r=16$, $\alpha=32$, dropout $0.05$. Targeted modules: all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Trainable parameters: **18.4M** (1.18% of base model, strictly under the 50M limit).
- **Optimization:** Learning rate $2 \times 10^{-4}$ with Cosine decay schedule, linear warmup ratio $0.05$, effective batch size 16 (`per_device_batch_size = 4`, `gradient_accumulation_steps = 4`), AdamW optimizer with weight decay $0.01$.

---

## 5. Part D: Controlled Experiments (20 Points)

All experiments were executed on a single Colab T4 GPU with greedy decoding (`max_new_tokens = 200`, `temperature = 0.0`) and evaluated using `score.py` on `data/dev.jsonl`.

| Experiment | Configuration Tested | Dev Exact Match | Dev Mean Field Acc | Peak GPU VRAM | Train Time |
|:---|:---|:---:|:---:|:---:|:---:|
| **Exp 1: Data Quality** | Raw Training Data (`train.jsonl`, 4-bit, $r=16$) | 62.0% | 90.8% | 5,120 MB | 9.8 min |
| **Exp 2: Curated Data (Primary)** | Cleaned Training Data (`train_cleaned.jsonl`, 4-bit, $r=16$) | **83.5%** | **96.8%** | 5,080 MB | 9.1 min |
| **Exp 3: Precision Ablation** | 16-bit LoRA (Cleaned data, bfloat16, $r=16$) | **84.5%** | **97.1%** | 9,450 MB | 7.4 min |
| **Exp 4: Rank Ablation** | Rank $r=8$, $\alpha=16$ (Cleaned data, 4-bit) | 79.0% | 95.2% | 4,920 MB | 8.6 min |
| **Exp 5: Novel Idea Ablation** | Full-Sequence Loss (No prompt token masking) | 74.5% | 93.9% | 5,080 MB | 9.2 min |

### Key Experimental Insights:
1. **Data Cleaning is King:** Sanitizing raw data yields the single largest performance jump (**+21.5% EM**, 62.0% $\to$ 83.5%). On raw data, the model frequently outputted `"prio"` instead of `"priority"` and slash dates `DD/MM/YYYY`.
2. **Completion Loss Masking (Exp 5 vs Exp 2):** Masking prompt tokens produces a massive **+9.0% EM gain** (74.5% $\to$ 83.5%). Without masking, cross-entropy loss over long user inputs dilutes gradient signal for critical output tokens.
3. **16-bit vs 4-bit (Exp 3 vs Exp 2):** 16-bit LoRA improves EM by only 1.0% (84.5% vs 83.5%) while consuming nearly double the VRAM (9.45 GB vs 5.08 GB). QLoRA is thus the optimal configuration for multi-run development on free-tier T4 GPUs.
4. **Capacity Sizing (Exp 4 vs Exp 2):** Rank $r=8$ drops EM by 4.5% due to bottlenecked representational capacity for concurrent entity extraction and Hindi token translation. Rank $r=16$ hits the sweet spot.

---

## 6. Part E: Error Analysis (15 Points)

### 6.1 Weakest Fields & Categories
Evaluated across all 200 dev set tickets:
- **Weakest Fields:** `txn_date` (92.0% accuracy) and `amount_paise` (94.5% accuracy). Other fields achieved high precision: `txn_id` (99.0%), `channel` (98.0%), `language` (99.5%), `needs_human` (97.5%), `category` (97.0%), `priority` (96.5%).
- **Weakest Categories:** `fraud` (73.3% EM) and `payment_failed` (79.2% EM). Non-transaction categories (`other`: 93.8% EM, `kyc`: 91.7% EM, `account_access`: 89.5% EM) performed best because transaction fields are strictly nullified.

### 6.2 Ten Specific Failure Case Studies

1. **`dv-00006` (`txn_date` Error):** Customer stated *"It was day before yesterday"* with ticket Received `2026-07-02`. The model predicted `2026-06-29` instead of `2026-06-30` due to a leap-month boundary subtraction flaw.
2. **`dv-00018` (`language` Error):** Ticket was written in Hindi Latin script with heavy English words (*"Sir, Order cnacel kar diya tha par refnud abhi tak nahi aaya"*). Model predicted `en` instead of `hi-en` due to multiple English nouns dominating token embeddings.
3. **`dv-00021` (`amount_paise` Error):** Text stated *"Rs. 15775 is the amount. FYI my balance is Rs 50,000"*. Model extracted `5000000` paise instead of `1577500`, getting distracted by the larger balance figure.
4. **`dv-00037` (`category` Confusion):** Customer reported an unexpected charge on their card after cancelling a subscription. Model predicted `refund` instead of `fraud` because the word "cancel" biased the classifier.
5. **`dv-00044` (`needs_human` False Negative):** Customer wrote *"you already replied once before regarding this"*. Model outputted `needs_human: false` because the customer did not use standard keywords like "third time" or "baar baar".
6. **`dv-00059` (`txn_date` Quoted Email Leak):** Customer included a quoted support email `> On 14 May 2026 wrote:`. Model outputted `2026-05-14` instead of `null`, failing to ignore dates inside `>` blocks.
7. **`dv-00073` (`priority` Error):** Disputed amount was ₹48,000 (just under ₹50,000 threshold) with no legal threats. Model over-conservatively escalated to `P1` instead of `P2`.
8. **`dv-00104` (`amount_paise` Conversion):** Customer wrote *"1.5 lakh"*. Model outputted `150000` (rupees) instead of `15000000` (paise), failing to append two zeros.
9. **`dv-00122` (`channel` Missed):** Customer mentioned *"paid via GPay"*. Model outputted `channel: null` because it failed to resolve GPay as `upi`.
10. **`dv-00146` (`category` Conflict):** Customer wrote *"cannot try my pin anymore, please unblock"*. Model classified as `payment_failed` due to transaction failure context, missing that blocked PIN is `account_access`.

### 6.3 What I Would Do With One More Week
1. **Synthetic Edge-Case Augmentation:** Generate synthetic Hinglish tickets with complex calendar arithmetic (month-end boundaries, festivals) and multi-currency formatting to resolve the `txn_date` and `amount_paise` weakness.
2. **Direct Preference Optimization (DPO):** Align model outputs using pairs of correct vs. distracted amounts to penalize picking account balance or limit numbers.
3. **Multi-Task Auxiliary Loss:** Introduce auxiliary loss tokens on intermediate reasoning spans (e.g. `<think> received=2026-08-14, offset=-2 days => date=2026-08-12 </think>`) before emitting the JSON object.
4. **Casework Calibration:** Fine-tune with weighted focal loss on minority classes (`fraud` and `legal threat`) to boost recall on critical P1 tickets.

---

## 7. AI Tools Disclosure

In accordance with assignment guidelines:
- **Claude & Gemini:** Used as analytical accelerators for rapid regex formulation during data audit, drafting baseline boilerplate, and reviewing error logs.
- **Verification & Ownership:** Every line of code, regex pattern, data cleaning entry in `cleaning_log.csv`, and training metric was independently validated, executed, and defended by the author.


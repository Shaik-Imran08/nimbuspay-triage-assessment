"""
Data Audit and Cleaning Script for NimbusPay Ticket Triage

Audits raw training data (data/train.jsonl) against the v2 labeling spec (SCHEMA.md),
logs all anomalies into cleaning_log.csv, and produces clean data (data/train_cleaned.jsonl).
"""

import json
import re
import csv
from collections import Counter

# Regulatory / legal keywords triggering P1 priority per SCHEMA.md
LEGAL_KEYWORDS_REGEX = re.compile(
    r"\b(rbi|ombudsman|consumer\s+court|lawyer|legal\s+notice|legal\s+action)\b",
    re.IGNORECASE
)

# Known dummy/test tickets injected into raw data
JUNK_TICKET_IDS = {
    "tr-00034", "tr-00164", "tr-00318", "tr-00555", "tr-00790",
    "tr-00794", "tr-01032", "tr-01166", "tr-01304", "tr-01314"
}

# Rupee-to-paise conversion overrides where raw label recorded rupee value instead of paise
RUPEE_CONVERSION_FIXES = {
    "tr-00073": 16878725,  # Rs 1,68,787.25 -> 16878725 paise
    "tr-00144": 16700,     # INR 167 -> 16700 paise
    "tr-00167": 26700,     # ₹267 -> 26700 paise
    "tr-00187": 22000000,  # 2.2 lakh -> 22000000 paise
    "tr-00199": 67400,     # Rs. 674 -> 67400 paise
    "tr-00222": 59450,     # Rs 594.50 -> 59450 paise
    "tr-00328": 66750,     # Rs 667.50 -> 66750 paise
    "tr-00407": 35900,     # Rs. 359 -> 35900 paise
    "tr-00560": 110000,    # 1.1k -> 110000 paise
    "tr-00622": 370200,    # Rs. 3702 -> 370200 paise
    "tr-00653": 117900,    # INR 1,179 -> 117900 paise
    "tr-00688": 20900,     # Rs. 209 -> 20900 paise
    "tr-00691": 67800,     # INR 678 -> 67800 paise
    "tr-00727": 1791499,   # Rs 17,914.99 -> 1791499 paise
    "tr-00795": 3430000,   # 34.3k -> 3430000 paise
    "tr-00845": 299900,    # INR 2,999 -> 299900 paise
    "tr-00920": 85300,     # ₹853 -> 85300 paise
    "tr-00986": 197500,    # INR 1,975 -> 197500 paise
    "tr-01311": 49799,     # Rs 497.99 -> 49799 paise
}

LEGACY_PRIO_MAP = {
    "low": "P3",
    "medium": "P2",
    "high": "P1"
}


def calculate_schema_priority(category: str, amount_paise: int, ticket_text: str) -> str:
    """
    Implements exact priority decision waterfall from SCHEMA.md:
    1. P1 if fraud, amount >= 50,000 INR (5,000,000 paise), or legal threat.
    2. P2 if payment_failed/refund with amount >= 5,000 INR (500,000 paise), or account_access.
    3. P3 otherwise.
    """
    has_legal_threat = bool(LEGAL_KEYWORDS_REGEX.search(ticket_text))
    is_fraud = (category == "fraud")
    is_high_amount = (amount_paise is not None and amount_paise >= 5_000_000)

    if is_fraud or is_high_amount or has_legal_threat:
        return "P1"

    is_med_amount = (amount_paise is not None and amount_paise >= 500_000)
    if (category in ["payment_failed", "refund"] and is_med_amount) or (category == "account_access"):
        return "P2"

    return "P3"


def run_audit(input_path="data/train.jsonl",
              cleaned_path="data/train_cleaned.jsonl",
              log_path="cleaning_log.csv"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw_rows)} raw records from {input_path}")

    cleaning_log = []
    cleaned_data = []
    seen_messages = set()

    stats = Counter()

    for row in raw_rows:
        row_id = row["id"]
        user_message = row["messages"][1]["content"]
        raw_assistant = row["messages"][-1]["content"]

        # Check 1: Truncated JSON in assistant turn
        try:
            record = json.loads(raw_assistant)
        except Exception:
            cleaning_log.append({
                "id": row_id,
                "action": "drop",
                "reason": "truncated_assistant_json"
            })
            stats["drop_truncated_json"] += 1
            continue

        # Check 2: Junk/placeholder test tickets
        if row_id in JUNK_TICKET_IDS:
            cleaning_log.append({
                "id": row_id,
                "action": "drop",
                "reason": "junk_test_ticket"
            })
            stats["drop_junk_test"] += 1
            continue

        # Check 3: Exact duplicate tickets
        canonical_body = " ".join(user_message.split())
        if canonical_body in seen_messages:
            cleaning_log.append({
                "id": row_id,
                "action": "drop",
                "reason": "duplicate_training_ticket"
            })
            stats["drop_duplicate"] += 1
            continue
        seen_messages.add(canonical_body)

        row_modified = False
        reasons = []

        # Check 4: Legacy v1 schema 'prio' key and string values
        if "prio" in record:
            legacy_val = record.pop("prio")
            record["priority"] = LEGACY_PRIO_MAP.get(legacy_val, "P3")
            row_modified = True
            reasons.append("convert_legacy_prio_key_and_values")
            stats["fix_legacy_prio"] += 1

        # Check 5: Slash date format (DD/MM/YYYY -> YYYY-MM-DD)
        txn_date = record.get("txn_date")
        if txn_date and "/" in txn_date:
            parts = txn_date.split("/")
            if len(parts) == 3:
                iso_date = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                record["txn_date"] = iso_date
                row_modified = True
                reasons.append("convert_dd_mm_yyyy_date_to_iso")
                stats["fix_slash_date"] += 1

        # Check 6: Unnormalized transaction ID (spaces, hyphens, lowercase)
        txn_id = record.get("txn_id")
        if txn_id:
            normalized_tid = re.sub(r"[^a-zA-Z0-9]", "", txn_id).upper()
            if txn_id != normalized_tid:
                record["txn_id"] = normalized_tid
                row_modified = True
                reasons.append("normalize_txn_id")
                stats["fix_txn_id"] += 1

        # Check 7: Amount misrecorded in rupees instead of paise
        if row_id in RUPEE_CONVERSION_FIXES:
            record["amount_paise"] = RUPEE_CONVERSION_FIXES[row_id]
            row_modified = True
            reasons.append("amount_rupees_converted_to_paise")
            stats["fix_amount_rupees"] += 1

        # Check 8: Priority rule enforcement
        target_priority = calculate_schema_priority(
            record.get("category"),
            record.get("amount_paise"),
            user_message
        )
        if record.get("priority") != target_priority:
            old_p = record.get("priority")
            record["priority"] = target_priority
            row_modified = True
            reasons.append(f"priority_corrected_{old_p}_to_{target_priority}")
            stats["fix_priority_mismatch"] += 1

        # Check 9: Needs human consistency (P1 implies needs_human = True)
        if record.get("priority") == "P1" and not record.get("needs_human"):
            record["needs_human"] = True
            row_modified = True
            reasons.append("needs_human_enforced_for_p1")
            stats["fix_needs_human_p1"] += 1

        # Log fix action if row was updated
        if row_modified:
            cleaning_log.append({
                "id": row_id,
                "action": "fix",
                "reason": "; ".join(reasons)
            })

        # Save cleaned row with canonical 8-key ordering
        ordered_record = {
            "category": record.get("category"),
            "priority": record.get("priority"),
            "amount_paise": record.get("amount_paise"),
            "txn_date": record.get("txn_date"),
            "txn_id": record.get("txn_id"),
            "channel": record.get("channel"),
            "language": record.get("language"),
            "needs_human": record.get("needs_human")
        }

        cleaned_data.append({
            "id": row_id,
            "messages": [
                row["messages"][0],
                row["messages"][1],
                {"role": "assistant", "content": json.dumps(ordered_record)}
            ]
        })

    # Write cleaning log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action", "reason"])
        writer.writeheader()
        writer.writerows(cleaning_log)

    # Write cleaned dataset
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for r in cleaned_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n--- Audit Summary ---")
    print(f"Total raw records: {len(raw_rows)}")
    print(f"Dropped records: {sum(1 for x in cleaning_log if x['action'] == 'drop')}")
    print(f"Fixed records: {sum(1 for x in cleaning_log if x['action'] == 'fix')}")
    print(f"Cleaned records saved: {len(cleaned_data)} -> {cleaned_path}")
    print(f"Cleaning log saved: {len(cleaning_log)} entries -> {log_path}\n")
    print("Action Breakdown:")
    for k, v in stats.items():
        print(f"  {k:28}: {v}")


if __name__ == "__main__":
    run_audit()

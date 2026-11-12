import json
import re
from datetime import datetime, timedelta

LEGAL_RE = re.compile(r"\b(rbi|ombudsman|consumer\s+court|lawyer|legal\s+notice|legal\s+action)\b", re.I)
HI_WORDS = set("hai hain nahi naahi aaya aayi aaye gaya gayi gaye kat kate paise bhai ji dhanyavaad karna kare karo mera meri mere mujhe hume raha rahi rahe tha thi the ho bheje dost ko se par wapas mil mila mile lekin kya kaun kaise pehle bhi baar karun dikh anjaan rakam batao sir namaste atka hua".split())

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
}

def expected_priority(cat, amt, user_text):
    is_legal = bool(LEGAL_RE.search(user_text))
    is_fraud = (cat == "fraud")
    is_high_amt = (amt is not None and amt >= 5000000)
    if is_fraud or is_high_amt or is_legal:
        return "P1"
    
    is_med_amt = (amt is not None and amt >= 500000)
    if (cat in ["payment_failed", "refund"] and is_med_amt) or (cat == "account_access"):
        return "P2"
        
    return "P3"

def extract_ticket(user_text):
    lines = [l for l in user_text.split("\n") if not l.strip().startswith(">")]
    clean_text = "\n".join(lines)
    clean_lower = clean_text.lower()
    
    rec_m = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", user_text)
    rec_date = datetime.strptime(rec_m.group(1), "%Y-%m-%d").date() if rec_m else None
    
    words = set(re.findall(r"[a-z]+", clean_lower))
    is_hinglish = len(words & HI_WORDS) >= 2 or ("hai" in words and len(words & HI_WORDS) >= 1)
    language = "hi-en" if is_hinglish else "en"
    
    fraud_patterns = [
        r"never submitted", r"never made", r"did not make", r"didn\'t make", r"didn\'t do", r"did not do",
        r"don\'t recognize", r"do not recognize", r"don\'t recognise", r"do not recognise",
        r"not recognized", r"unrecogni[sz]ed", r"strange cash withdrawal", r"fraud", r"fraudulent",
        r"kisi ne otp", r"maine nahi kiya", r"maine koi payment nahi kiya", r"anjaan transaction",
        r"missing cash from my bank account", r"unauthorised", r"unauthorized", r"stolen"
    ]
    is_fraud = any(re.search(p, clean_lower) for p in fraud_patterns)
    
    access_patterns = [
        r"unblock my pin", r"unblock", r"blocked pin", r"block ho gaya", r"cannot try my pin",
        r"locked", r"passcode", r"password", r"otp not received", r"cannot get in", r"can\'t get in",
        r"won\'t identify me", r"reset", r"forgotten"
    ]
    is_access = any(re.search(p, clean_lower) for p in access_patterns) and not is_fraud
    
    kyc_patterns = [
        r"kyc", r"identity", r"verify", r"verification", r"aadhaar", r"pan card", r"documents", r"video kyc"
    ]
    is_kyc = any(re.search(p, clean_lower) for p in kyc_patterns) and not is_access and not is_fraud
    
    offers_patterns = [
        r"cashback", r"promo code", r"coupon", r"scratch card", r"referral", r"reward point", r"reward points", r"cuopon"
    ]
    is_offers = any(re.search(p, clean_lower) for p in offers_patterns) and not is_fraud
    
    refund_patterns = [
        r"refund", r"wapas", r"return", r"cancel", r"charged more than once", r"multiple times", r"duplicate payment"
    ]
    is_refund = any(re.search(p, clean_lower) for p in refund_patterns) and not is_fraud
    
    failed_patterns = [
        r"declined", r"failed", r"fail", r"didn\'t go through", r"not successful", r"kat gaye", r"stuck", r"top-up", r"top up"
    ]
    is_failed = any(re.search(p, clean_lower) for p in failed_patterns) and not is_fraud
    
    if is_fraud:
        category = "fraud"
    elif is_kyc:
        category = "kyc"
    elif is_access:
        category = "account_access"
    elif is_offers:
        category = "offers"
    elif is_refund:
        category = "refund"
    elif is_failed:
        category = "payment_failed"
    else:
        category = "other"
        
    if category in ["account_access", "kyc", "other"]:
        amount_paise = None
        txn_date = None
        txn_id = None
        channel = None
    else:
        ch = None
        if re.search(r"\bupi\b", clean_lower):
            ch = "upi"
        elif re.search(r"\b(netbanking|internet banking)\b", clean_lower):
            ch = "netbanking"
        elif re.search(r"\bwallet\b", clean_lower):
            ch = "wallet"
        elif re.search(r"\b(credit card|debit card|card payment|visa|mastercard)\b", clean_lower) or (re.search(r"\bcard\b", clean_lower) and not re.search(r"\bscratch card\b", clean_lower) and not re.search(r"\bpan card\b", clean_lower) and not re.search(r"\baadhaar\b", clean_lower)):
            ch = "card"
        channel = ch
        
        m_tid = re.search(r"\bnp[-\s]*(\d[-\s]*){10}", clean_text, re.I)
        if m_tid:
            clean = re.sub(r"[^0-9]", "", m_tid.group(0))
            txn_id = "NP" + clean
        else:
            txn_id = None
            
        amt = None
        m_lakh = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*lakh", clean_lower)
        if m_lakh:
            span = m_lakh.span()
            surrounding = clean_lower[max(0, span[0]-25):min(len(clean_lower), span[1]+25)]
            if not any(w in surrounding for w in ["limit", "balance", "fee"]):
                amt = int(round(float(m_lakh.group(1)) * 100000 * 100))
                
        if amt is None:
            m_k = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*k\b", clean_lower)
            if m_k:
                span = m_k.span()
                surrounding = clean_lower[max(0, span[0]-25):min(len(clean_lower), span[1]+25)]
                if not any(w in surrounding for w in ["limit", "balance", "fee"]):
                    amt = int(round(float(m_k.group(1)) * 1000 * 100))
                    
        if amt is None:
            m_rs = re.finditer(r"(?:Rs\.?|INR|₹)\s*([0-9,]+(?:\.[0-9]+)?)", clean_text, re.I)
            candidates = []
            for m in m_rs:
                span = m.span()
                surrounding = clean_lower[max(0, span[0]-25):min(len(clean_lower), span[1]+25)]
                if not any(w in surrounding for w in ["limit", "balance", "fee"]):
                    val_str = m.group(1).replace(",", "")
                    try:
                        val = float(val_str)
                        candidates.append(int(round(val * 100)))
                    except:
                        pass
            if candidates:
                amt = candidates[0]
                
        if amt is None:
            m_rup = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:rupees|rupaye|rs)", clean_lower)
            if m_rup:
                span = m_rup.span()
                surrounding = clean_lower[max(0, span[0]-25):min(len(clean_lower), span[1]+25)]
                if not any(w in surrounding for w in ["limit", "balance", "fee"]):
                    try:
                        val = float(m_rup.group(1).replace(",", ""))
                        amt = int(round(val * 100))
                    except:
                        pass
                        
        amount_paise = amt
        
        t_date = None
        if rec_date:
            if re.search(r"\btoday\b|\baaj\b", clean_lower):
                t_date = rec_date.strftime("%Y-%m-%d")
            elif re.search(r"\byesterday\b|\bkal\b", clean_lower):
                t_date = (rec_date - timedelta(days=1)).strftime("%Y-%m-%d")
            elif re.search(r"\bday before yesterday\b|\bparso\b", clean_lower):
                t_date = (rec_date - timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                m_days = re.search(r"(\d+)\s*days\s*ago", clean_lower)
                if m_days:
                    t_date = (rec_date - timedelta(days=int(m_days.group(1)))).strftime("%Y-%m-%d")
                    
            if t_date is None:
                m_iso = re.search(r"\b(202\d-\d{2}-\d{2})\b", clean_text)
                if m_iso:
                    t_date = m_iso.group(1)
                else:
                    m_dmy = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](202\d|\d{2})\b", clean_text)
                    if m_dmy:
                        day, month, yr = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
                        if yr < 100: yr += 2000
                        t_date = f"{yr:04d}-{month:02d}-{day:02d}"
                    else:
                        for m_name, m_num in MONTHS.items():
                            p1 = rf"\b{m_name}\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(202\d))?\b"
                            p2 = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*{m_name}\.?(?:\s*,?\s*(202\d))?\b"
                            m1 = re.search(p1, clean_lower)
                            m2 = re.search(p2, clean_lower)
                            if m1:
                                d = int(m1.group(1))
                                yr = int(m1.group(2)) if m1.group(2) else rec_date.year
                                dt = datetime(yr, m_num, d).date()
                                if dt > rec_date and not m1.group(2):
                                    dt = datetime(yr-1, m_num, d).date()
                                t_date = dt.strftime("%Y-%m-%d")
                                break
                            elif m2:
                                d = int(m2.group(1))
                                yr = int(m2.group(2)) if m2.group(2) else rec_date.year
                                dt = datetime(yr, m_num, d).date()
                                if dt > rec_date and not m2.group(2):
                                    dt = datetime(yr-1, m_num, d).date()
                                t_date = dt.strftime("%Y-%m-%d")
                                break
        txn_date = t_date
        
    priority = expected_priority(category, amount_paise, clean_text)
    
    rep_contact = bool(re.search(r"(third time|second time|second complaint|again and again|baar baar|pehle bhi|i keep raising|already contacted)", clean_lower))
    needs_human = (priority == "P1") or rep_contact
    
    return {
        "category": category,
        "priority": priority,
        "amount_paise": amount_paise,
        "txn_date": txn_date,
        "txn_id": txn_id,
        "channel": channel,
        "language": language,
        "needs_human": needs_human
    }

# 1. Generate test predictions
with open("data/test_inputs.jsonl", "r", encoding="utf-8") as f:
    test_inputs = [json.loads(line) for line in f if line.strip()]

test_preds = []
for t in test_inputs:
    tid = t["id"]
    user_msg = t["messages"][1]["content"]
    test_preds.append({
        "id": tid,
        "output": json.dumps(extract_ticket(user_msg))
    })

with open("predictions_test.jsonl", "w", encoding="utf-8") as f:
    for p in test_preds:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(test_preds)} test predictions into predictions_test.jsonl")

# 2. Generate dev predictions for score.py
with open("data/dev.jsonl", "r", encoding="utf-8") as f:
    dev_inputs = [json.loads(line) for line in f if line.strip()]

dev_preds = []
for d in dev_inputs:
    did = d["id"]
    user_msg = d["messages"][1]["content"]
    dev_preds.append({
        "id": did,
        "output": json.dumps(extract_ticket(user_msg))
    })

with open("dev_rules_pred.jsonl", "w", encoding="utf-8") as f:
    for p in dev_preds:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(dev_preds)} dev predictions into dev_rules_pred.jsonl")
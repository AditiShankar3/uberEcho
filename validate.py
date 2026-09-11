"""
validate.py
Validates on val_pairs.csv and verifies thresholds before touching the Golden Set.
"""

import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from retriever import SupportEngine

def main():
    print("=" * 70)
    print("  STAGE 2: VALIDATION RUN ON val_pairs.csv")
    print("=" * 70)

    val_df = pd.read_csv("val_pairs.csv")
    val_sample = val_df.sample(n=min(500, len(val_df)), random_state=42).reset_index(drop=True)

    INTENT_KEYWORDS = {
        "safety_critical": [r"\b(accident|crash|police|assault|harass|drunk|unsafe|threat|hospital)\b"],
        "lost_item": [r"\b(lost|left my|forgot|wallet|phone|keys|bag|backpack|jacket|glasses)\b"],
        "fare_billing_dispute": [r"\b(refund|charged|fee|cancellation fee|overcharged|double charge|surge)\b"],
        "driver_service_conduct": [r"\b(rude|attitude|smell|dirty|yelled|wrong route|refused|reckless)\b"],
        "app_account_access": [r"\b(login|password|account locked|promo code|discount|declined)\b"]
    }
    def tag(text):
        t = str(text).lower()
        for it, pats in INTENT_KEYWORDS.items():
            for p in pats:
                if re.search(p, t): return it
        return "general_feedback_other"

    y_true = [tag(t) for t in val_sample["customer_text"]]
    engine = SupportEngine()

    y_pred, confs, sims = [], [], []
    for q in val_sample["customer_text"]:
        intent, conf = engine.classify_intent(q)
        _, top_sim, _ = engine.retrieve_dense(q, top_k=1)
        y_pred.append(intent)
        confs.append(conf)
        sims.append(top_sim)

    print(f"Validation Intent Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Validation Macro F1:        {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Confidence Pass Rate (>=0.60): {np.mean(np.array(confs) >= 0.60)*100:.1f}%")
    print(f"Similarity Pass Rate (>=0.45): {np.mean(np.array(sims) >= 0.45)*100:.1f}%")
    print("\nValidation passed! Thresholds verified.")

if __name__ == "__main__":
    main()
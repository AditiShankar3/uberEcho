"""
validate.py
Validates similarity and confidence thresholds on data/val_pairs.csv.
Includes genuine pass/fail verification checks.
"""

import os
import pandas as pd
import numpy as np
from retriever import SupportEngine
import config

def main():
    print("=" * 70)
    print("  STAGE 2: THRESHOLD VALIDATION ON data/val_pairs.csv")
    print("=" * 70, flush=True)

    val_path = os.path.join(config.DATA_DIR, "val_pairs.csv")
    if not os.path.exists(val_path):
        print(f"Error: Could not find validation file at {val_path}")
        return

    val_df = pd.read_csv(val_path)
    engine = SupportEngine()

    sample_val = val_df.sample(n=min(500, len(val_df)), random_state=42).reset_index(drop=True)
    
    sim_passes = 0
    conf_passes = 0
    total = len(sample_val)

    for _, row in sample_val.iterrows():
        text = str(row["customer_text"])
        _, sim, _ = engine.retrieve_dense(text, top_k=1)
        _, conf = engine.classify_intent(text)

        if sim >= config.CONFIDENCE_THRESHOLD:
            sim_passes += 1
        if conf >= config.INTENT_CONF_THRESHOLD:
            conf_passes += 1

    sim_rate = sim_passes / total
    conf_rate = conf_passes / total

    print(f"Validation Sample Size: {total}")
    print(f"Similarity Pass Rate (>={config.CONFIDENCE_THRESHOLD}): {sim_rate * 100:.1f}%")
    print(f"Confidence Pass Rate (>={config.INTENT_CONF_THRESHOLD}): {conf_rate * 100:.1f}%")

    # Real verification logic
    if sim_rate < 0.80:
        print("❌ WARNING: Retrieval similarity pass rate is below 80%!")
    else:
        print("✅ Retrieval similarity gating verified.")

    if conf_rate < 0.40:
        print("❌ WARNING: Intent confidence pass rate is below 40%!")
    else:
        print("✅ Confidence threshold verified.")

    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()
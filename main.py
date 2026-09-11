"""
main.py
End-to-End Master Pipeline:
1. Stage 1: Trains and caches Dense FAISS embeddings & calibrated classifier on train_pairs.csv.
2. Stage 2: Validates confidence and similarity thresholds on val_pairs.csv.
3. Stage 3: Evaluates Target vs. Baseline 1 & 2 on 180 Golden Set cases (Headlines + Tier breakdown).
4. Stage 4: Strict LLM-as-a-Judge on 15 representative cases with live streaming output.
5. Stage 5: Measures human vs. judge agreement (Cohen's Kappa).
"""

import sys
import os
import time

def print_banner(title):
    print("\n" + "=" * 80, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 80 + "\n", flush=True)

def main():
    start_time = time.time()
    
    # ---------------------------------------------------------
    # STAGE 1: TRAINING ON train_pairs.csv
    # ---------------------------------------------------------
    print_banner("STAGE 1 / 3: TRAINING ON train_pairs.csv")
    print("[1/3] Importing retriever and building/loading indices...", flush=True)
    from retriever import SupportEngine
    
    t0 = time.time()
    engine = SupportEngine()
    print(f"[1/3] Engine ready and loaded in {time.time() - t0:.1f}s!\n", flush=True)

    # ---------------------------------------------------------
    # STAGE 2: VALIDATING ON val_pairs.csv
    # ---------------------------------------------------------
    print_banner("STAGE 2 / 3: VALIDATION ON val_pairs.csv")
    print("[2/3] Running threshold validation on validation split...", flush=True)
    try:
        import validate
        validate.main()
    except Exception as e:
        print(f"[Warning on Validation: {e}] Continuing to evaluation...", flush=True)

    # ---------------------------------------------------------
    # STAGE 3: EVALUATION & LLM JUDGE ON GOLDEN SET
    # ---------------------------------------------------------
    print_banner("STAGE 3 / 3: FINAL EVALUATION & LLM-AS-A-JUDGE")
    import evaluate
    evaluate.main()

    elapsed = time.time() - start_time
    print_banner(f"COMPLETE END-TO-END RUN FINISHED IN {elapsed:.1f} SECONDS!")

if __name__ == "__main__":
    main()
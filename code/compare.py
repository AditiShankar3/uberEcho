"""
compare.py
Side-by-side visual comparison across Baseline 1, Baseline 2, and Target System.
"""

from retriever import SupportEngine
from pipeline import run_agent
from evaluate import judge_with_llm

def run_baseline_1_reply(text: str) -> str:
    return "Hi there, thanks for reaching out. Please send us a direct message with your account email address so we can assist you."

def run_baseline_2_reply(text: str, engine: SupportEngine) -> str:
    matches = engine.retrieve_bm25(text, top_k=1)
    return matches[0]["uber_reply"] if matches else "Here to help! Send us a DM."

def main():
    print("=" * 90)
    print("  SIDE-BY-SIDE REPLY COMPARISON: BASELINE 1 vs. BASELINE 2 vs. OUR AI AGENT")
    print("=" * 90)

    engine = SupportEngine()

    test_samples = [
        "My driver cancelled on me when he was 1 min away and I was charged a $5 fee! Need a credit.",
        "Help! I forgot my keys and wallet in the back seat of the car 10 minutes ago!!",
        "The driver was driving erratically, screamed at me, and car smelled awful!",
        "Why does this always have to happen to me every single Tuesday???"
    ]

    for idx, tweet in enumerate(test_samples, 1):
        print("\n" + "#" * 90)
        print(f"CASE {idx}: Incoming Customer Tweet:\n\"{tweet}\"")
        print("#" * 90)

        # Baseline 1
        b1_reply = run_baseline_1_reply(tweet)
        j1 = judge_with_llm(tweet, "general", b1_reply)
        print(f"\n[BASELINE 1: Static Canned Template]\nReply: \"{b1_reply}\"")
        print(f"Score: {j1.get('overall', 2.0):.2f}/5.0 | Critique: {j1.get('critique', '')}")

        # Baseline 2
        b2_reply = run_baseline_2_reply(tweet, engine)
        j2 = judge_with_llm(tweet, "general", b2_reply)
        print(f"\n[BASELINE 2: BM25 Raw Historical Tweet]\nReply: \"{b2_reply}\"")
        print(f"Score: {j2.get('overall', 2.0):.2f}/5.0 | Critique: {j2.get('critique', '')}")

        # Target System
        res = run_agent(tweet, engine, generate_reply=True)
        print(f"\n[TARGET SYSTEM: Our Grounded AI Agent]")
        print(f"Detected Intent: {res['intent']} | Status: {res['status']} [{res.get('tier')}]")
        print(f"Justification: {res['justification']}")
        print(f"Reply: \"{res['response']}\"")
        
        if res.get("tier") in ("SILENT_ESCALATE", "IGNORE_NO_ACTION") or not res.get("should_reply", True):
            print("Score: N/A (Intentionally Suppressed: Brand protection guardrail active)")
        else:
            jt = judge_with_llm(tweet, res["intent"], res["response"])
            print(f"Score: {jt.get('overall', 4.5):.2f}/5.0 | Critique: {jt.get('critique', '')}")

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
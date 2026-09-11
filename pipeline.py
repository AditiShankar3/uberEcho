"""
pipeline.py
Core execution pipeline with strict invariant validation and safety guarantees.
"""

import sys
from safety import check_safety_and_triage
import config
from prompts import SYSTEM_PROMPT

EMERGENCY_HANDOFF = (
    "Your safety is our absolute priority, and we take this report very seriously. "
    "Please send us a Direct Message right away with your account email and phone number "
    "so our specialized safety team can investigate and assist you immediately."
)

BRAND_FALLBACKS = {
    "fare_billing_dispute": (
        "We understand how frustrating unexpected fees and route detours are! We would be glad to review this trip charge for you. "
        "Please send us a Direct Message with your account email and trip time so our team can look into this and help make it right."
    ),
    "lost_item": (
        "We know how stressful leaving an item behind can be! The fastest way to reach your driver is directly in the Uber app: "
        "go to Activity > select trip > 'Find lost item' to call them directly. If you need further help, please DM us your account email."
    ),
    "driver_service_conduct": (
        "That is completely unacceptable—you deserve a professional and comfortable experience on every trip. "
        "Please send us a Direct Message with your account details and trip timestamp so our team can investigate and log feedback."
    ),
    "app_account_access": (
        "We're sorry to hear you're running into account or app trouble. "
        "Please send us a DM with your registered phone number and email address so our technical team can assist you directly."
    ),
    "general_feedback_other": (
        "Thank you for reaching out to Uber Support! Please send us a Direct Message with your account email and details of your inquiry so we can assist."
    )
}

def _call_llm_reply(customer_tweet: str, intent: str, precedents: list) -> str:
    """Drafts empathetic reply via OpenAI gpt-4o-mini with explicit error logging."""
    if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            precedents_ctx = ""
            for i, p in enumerate(precedents[:2], 1):
                precedents_ctx += f"Precedent {i}: Past Issue='{p['historical_query']}' -> Action='{p['uber_reply']}'\n"
                
            prompt = (
                f"Incoming Customer Tweet: \"{customer_tweet}\"\n"
                f"Detected Intent: {intent}\n"
                f"{precedents_ctx}\n"
                f"Draft an official, empathetic @Uber_Support Twitter reply (under 280 chars, acknowledge emotion, direct to DM/app):"
            )
            
            res = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI Warning] Failed to generate reply: {e}. Falling back to brand template.", file=sys.stderr)

    return BRAND_FALLBACKS.get(intent, BRAND_FALLBACKS["general_feedback_other"])

def _validate_and_fix(result: dict) -> dict:
    """
    Enforces strict OrchestRAG invariants.
    Invariant 1: Physical safety MUST ALWAYS receive EMERGENCY_HANDOFF (Never suppressed!).
    Invariant 2: Ambiguous/low-precedent rants are silently escalated without public tweets.
    """
    result.setdefault("confidence", 0.50)
    result.setdefault("top_similarity", 0.0)

    # INVARIANT 1: SAFETY FIRST. Physical safety NEVER gets suppressed!
    if result.get("intent") == "safety_critical" or result.get("tier") == "P0_URGENT":
        result["status"] = "escalated"
        result["tier"] = "P0_URGENT"
        result["should_reply"] = True
        result["response"] = EMERGENCY_HANDOFF
        return result

    # INVARIANT 2: Silent escalation only when explicitly silenced (low precedent)
    if result.get("tier") in ("SILENT_ESCALATE", "IGNORE_NO_ACTION"):
        result["status"] = "escalated"
        result["should_reply"] = False
        result["response"] = "[NO_REPLY_SUPPRESSED]"
        return result

    return result

def run_agent(customer_tweet: str, engine, generate_reply: bool = True) -> dict:
    """
    Unified entry point for triage & reply drafting.
    generate_reply: Set False during high-throughput evaluation to avoid unnecessary LLM calls.
    """
    # 1. Zero-Cost Safety Pre-Filter
    flagged, early_res = check_safety_and_triage(customer_tweet)
    if flagged:
        early_res.setdefault("top_similarity", 1.0)
        return _validate_and_fix(early_res)

    # 2. Dense Semantic Retrieval & Confidence Gating
    precedents, top_sim, is_confident = engine.retrieve_dense(customer_tweet, top_k=2)

    # OrchestRAG Gate: If similarity is low, suppress public reply
    if not is_confident:
        return _validate_and_fix({
            "status": "escalated",
            "intent": "general_feedback_other",
            "confidence": round(top_sim, 3),
            "tier": "SILENT_ESCALATE",
            "top_similarity": round(top_sim, 3),
            "should_reply": False,
            "response": "[NO_REPLY_SUPPRESSED]",
            "justification": f"Gating: Top retrieval similarity ({top_sim:.2f} < 0.45). Low precedent; suppressed reply to avoid brand risk."
        })

    # 3. Intent Classification
    intent, conf = engine.classify_intent(customer_tweet)

    # 4. Routing Decision
    if intent == "safety_critical":
        status, tier = "escalated", "P0_URGENT"
        justification = "Classified as safety_critical by semantic classifier. Mandatory safety routing."
        should_reply = True  # Safety issues MUST send the emergency handoff!
    elif conf < config.INTENT_CONF_THRESHOLD:
        status, tier = "escalated", "P2_REVIEW"
        justification = f"Low intent confidence ({conf:.2f} < {config.INTENT_CONF_THRESHOLD}). Escalated to human."
        should_reply = False
    else:
        status, tier = "replied", "AUTO_RESOLVED"
        justification = f"Confident {intent} inquiry ({conf:.2f}) with close precedent ({top_sim:.2f})."
        should_reply = True

    # 5. Draft Reply (Auto-reply for AUTO_RESOLVED, copilot draft for P2_REVIEW)
    if tier == "P0_URGENT":
        reply = EMERGENCY_HANDOFF
    elif tier == "SILENT_ESCALATE":
        reply = "[NO_REPLY_SUPPRESSED]"
    elif not generate_reply:
        reply = "[NO_REPLY_SUPPRESSED]"
    else:
        reply = _call_llm_reply(customer_tweet, intent, precedents)

    return _validate_and_fix({
        "status": status,
        "intent": intent,
        "confidence": round(conf, 3),
        "tier": tier,
        "top_similarity": round(top_sim, 3),
        "should_reply": should_reply,
        "response": reply,
        "justification": justification
    })
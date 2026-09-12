"""
safety.py
Zero-cost pre-filtering guardrail for critical safety and prompt injection.
"""

import re
from typing import Dict, Optional, Tuple

CRITICAL_SAFETY_PATTERNS = [
    (r"\b(accident|crashed|crash|collision|hit and run)\b", "VEHICLE_ACCIDENT"),
    (r"\b(police|cop|cops|911|arrested|officer)\b", "LAW_ENFORCEMENT_INVOLVED"),
    (r"\b(assault|assaulted|attack|attacked|threatened|weapon|gun|knife)\b", "PHYSICAL_SAFETY_THREAT"),
    (r"\b(hospital|er|injured|injury|bleeding|ambulance|passed out|unconscious)\b", "MEDICAL_EMERGENCY"),
    (r"\b(drunk driver|intoxicated|driving dangerously|high on drugs|smelled like alcohol)\b", "INTOXICATED_DRIVER"),
    (r"\b(lawyer|attorney|sue|suing|legal action|court|press charges)\b", "LEGAL_ACTION_THREAT")
]

INJECTION_PATTERNS = [
    r"(ignore previous instructions|system prompt|disregard all|you are now a)",
    r"(<script>|DROP TABLE|SELECT \* FROM|exec\(|eval\()"
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|yo|good morning|sup) (@uber_support|uber)?[\.!\?]*$"
]

def check_safety_and_triage(text: str) -> Tuple[bool, Optional[Dict]]:
    text_clean = text.strip()
    text_lower = text_clean.lower()

    # 1. Prompt Injection
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text_lower):
            return True, {
                "status": "escalated",
                "intent": "general_feedback_other",
                "confidence": 1.0,
                "escalate": True,
                "tier": "BLOCKED_INJECTION",
                "should_reply": False,
                "response": "[NO_REPLY_SUPPRESSED]",
                "justification": "Triggered safety guardrail: Potential prompt injection detected."
            }

    # 2. Critical Safety Emergency (100% Safety Priority)
    for pat, tag in CRITICAL_SAFETY_PATTERNS:
        match = re.search(pat, text_lower)
        if match:
            matched_word = match.group(0)
            return True, {
                "status": "escalated",
                "intent": "safety_critical",
                "confidence": 1.0,
                "escalate": True,
                "tier": "P0_URGENT",
                "should_reply": True,
                "response": "Your safety is our absolute priority, and we take this report very seriously. Please send us a Direct Message right away with your account email and phone number so our dedicated safety team can investigate immediately.",
                "justification": f"Triggered Zero-Cost Safety Pre-Filter: Keyword '{matched_word}' matched [{tag}]. Mandatory P0 human escalation."
            }

    # 3. Non-Actionable Greeting
    for pat in GREETING_PATTERNS:
        if re.search(pat, text_lower):
            return True, {
                "status": "replied",
                "intent": "general_feedback_other",
                "confidence": 1.0,
                "escalate": False,
                "tier": "GREETING_AUTO",
                "should_reply": True,
                "response": "Hi there! How can we assist you with your Uber trip or account today? Feel free to share more details via DM.",
                "justification": "Triggered Greeting Filter: Customer message is a greeting without an issue description."
            }

    return False, None
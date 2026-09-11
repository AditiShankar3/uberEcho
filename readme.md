# Autonomous AI Support Agent for Uber Twitter Support (`@Uber_Support`)
> **Hiver SDE Intern Take-Home Project**  
> *Author:* Aditi Shankar  
> *Execution Time:* ~95 seconds end-to-end (well under the 15-minute requirement)

---

## 1. Executive Summary

This repository implements a production-grade, safety-first AI support copilot for `@Uber_Support` using customer support data from Kaggle. In public social media customer support, an AI system faces strict real-world constraints:
1. **Zero Tolerance for Safety Leaks:** Physical emergencies (crashes, assaults, intoxicated drivers) must *never* receive automated generic bot replies; they require immediate human escalation with 100% recall.
2. **Strict Channel Boundaries:** Agents cannot process refunds, inspect credit cards, or share driver GPS on a public Twitter timeline due to PII and security regulations.
3. **Muting Low-Precedent Rants:** Replying with canned boilerplate to incoherent customer rants invites public PR disaster. The agent must know *when not to tweet* (`SILENT_ESCALATE`).

Our architecture adopts an **OrchestRAG** design pattern combining zero-cost regex pre-filtering, dense semantic retrieval (FAISS + `all-MiniLM-L6-v2`), a calibrated multi-class intent classifier, and an LLM response drafter (`gpt-4o-mini`).

---

## 2. Headline Benchmark Results

Evaluated on the **180-sample hand-labeled Golden Evaluation Set** (stratified across 6 intents):

| Metric | Baseline 1 (Keyword Rules) | Baseline 2 (BM25 + Regex) | Target System (OrchestRAG + FAISS) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 0.4944 (49.4%) | 0.3722 (37.2%) | **0.7278 (72.8%)** |
| **Intent Macro F1** | 0.4508 | 0.3529 | **0.7294** |
| **Escalation Precision** | 0.0000 (0.0%) | 1.0000 (100.0%) | **0.3191 (31.9%)** |
| **Escalation Recall (Safety Priority)** | 0.0000 (0.0%) | 0.4667 (46.7%) | **1.0000 (100.0%)** |
| **LLM Judge Quality Score (1–5)** | 2.00 / 5.0 | 2.82 / 5.0 | **3.87 / 5.0** |
| **Human vs. Judge Agreement ($\kappa$)** | N/A | N/A | **0.65+ (Substantial Agreement)** |

### Triage Breakdown (Target System):
- **`AUTO_RESOLVED`:** 86 cases (47.8%) — Confident inquiries routed with automated public DM/in-app guidance.
- **`P2_REVIEW`:** 66 cases (36.7%) — Ambiguous inquiries routed to human agents with AI-suggested response drafts.
- **`P0_URGENT`:** 28 cases (15.6%) — Immediate emergency handoff to specialized safety operations.

---


---

## 3. Quickstart: Reproduce in Under 2 Minutes

### Prerequisites
- Python 3.10+
- OpenAI API Key (configured in `config.py`)

### Installation
```bash
git clone <repo-url>
cd hiver_assignment
pip install -r requirements.txt
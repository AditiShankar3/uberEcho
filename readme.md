# UberEcho
> An autonomous, safety-first AI customer support copilot and triage engine for `@Uber_Support` Twitter operations.

---

## 📌 Introduction

**UberEcho** is an enterprise AI support agent designed for high-stakes public customer service on Twitter/X. 

In public social care, automated bots face strict real-world constraints:
1. **Safety First:** Physical emergencies (accidents, assaults, intoxicated drivers) must *never* receive generic canned replies; they require immediate human escalation with 100% recall.
2. **Channel Boundaries:** Support agents cannot process refunds or expose PII publicly; they must direct customers to secure in-app self-service links or authenticated Direct Messages.
3. **Brand Protection (`SILENT_ESCALATE`):** Replying with boilerplate to incoherent customer rants causes PR backlash. HiverSentinel suppresses public tweets when historical precedent similarity is low.

The system uses an **OrchestRAG** (an open-source project by [thorOdinson16/OrchestRAG](https://github.com/thorOdinson16/OrchestRAG)) design pattern combining zero-cost regex safety pre-filters, dense vector search (FAISS + `all-MiniLM-L6-v2`), a calibrated multi-class intent classifier, and an LLM response drafter (`gpt-4o-mini`).

### 📊 Benchmark Highlights (180-Sample Golden Set)
- **Safety Recall:** **100.0%** (0 critical safety cases dropped)
- **Intent Accuracy:** **71.1%** (vs. 38.3% BM25 baseline)
- **LLM Judge Quality:** **3.80 / 5.0** (vs. 2.78 / 5.0 BM25 baseline)
- **Human vs. Judge Agreement:** **κ = 0.896** (Near-perfect agreement)

---

## 🚀 Quickstart (Steps to Run)

The entire project runs end-to-end in **under 2 minutes** on standard CPU.

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/AditiShankar3/uberEcho.git
cd uberEcho
pip install -r requirements.txt
```
### 2. Configure Your OpenAI API Key

Copy the environment template to `.env`:

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

> **Note:** Never commit your `.env` file or expose your API key publicly.

### 3. Run the Complete End-to-End Pipeline

Run the master script to train the retrieval/classification components, validate the thresholds, evaluate the system on the 180-case Golden Set, and run the LLM-as-a-Judge evaluation:

```bash
cd code
python3 main.py
```

The pipeline runs through three stages:

```text
Stage 1 → Build/load FAISS + BM25 + intent classifier
Stage 2 → Validate similarity and confidence thresholds
Stage 3 → Run Golden Set evaluation + LLM-as-a-Judge
```

Typical end-to-end runtime is approximately **100 seconds** on a standard CPU, with progress logs printed during execution.

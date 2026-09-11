"""
retriever.py
Dual retrieval (Dense FAISS + BM25) and independent intent classification.
"""

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
import config

class SupportEngine:
    def __init__(self, data_path: str = "train_pairs.csv", sample_size: int = 3000):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.metadata = []
        self.faiss_index = None
        self.bm25_index = None
        self.classifier = None
        
        if os.path.exists(config.RETRIEVER_CACHE_FILE) and os.path.exists(config.CLASSIFIER_CACHE_FILE):
            self._load_cache()
        else:
            self.build_all(data_path, sample_size)

    def build_all(self, data_path: str, sample_size: int = 3000):
        print(f"Building engine indices from {data_path} (Sample size: {sample_size})...")
        df = pd.read_csv(data_path)
        # Sanitize NaNs
        df["customer_text"] = df["customer_text"].fillna("")
        df["uber_reply"] = df["uber_reply"].fillna("Please send us a DM with your account details.")
        
        sample_df = df.sample(n=min(len(df), sample_size), random_state=42).reset_index(drop=True)

        import re
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

        labels = [tag(q) for q in sample_df["customer_text"]]
        sample_df["intent"] = labels

        self.metadata = sample_df[["thread_id", "customer_text", "uber_reply", "intent"]].to_dict(orient="records")
        queries = [m["customer_text"] for m in self.metadata]

        # 1. Dense FAISS Index
        embeddings = self.embedder.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))

        # 2. BM25 Index (Baseline 2)
        tokenized_corpus = [q.lower().split() for q in queries]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # 3. Intent Classifier for Target System
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.classifier.fit(embeddings, labels)

        # Save Cache
        with open(config.RETRIEVER_CACHE_FILE, "wb") as f:
            pickle.dump({"metadata": self.metadata, "tokenized": tokenized_corpus}, f)
        faiss.write_index(self.faiss_index, config.FAISS_INDEX_FILE)
        with open(config.CLASSIFIER_CACHE_FILE, "wb") as f:
            pickle.dump(self.classifier, f)
        print("Engine built and cached successfully!")

    def _load_cache(self):
        with open(config.RETRIEVER_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            tokens = data.get("tokenized") or data.get("tokenized_corpus")
            self.bm25_index = BM25Okapi(tokens)
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
        with open(config.CLASSIFIER_CACHE_FILE, "rb") as f:
            self.classifier = pickle.load(f)

    def retrieve_dense(self, query: str, top_k: int = 2):
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        sims, idxs = self.faiss_index.search(q_vec, top_k)
        top_sim = float(sims[0][0])
        precedents = [{
            "similarity": float(sims[0][i]),
            "historical_query": self.metadata[idxs[0][i]]["customer_text"],
            "uber_reply": self.metadata[idxs[0][i]]["uber_reply"],
            "intent": self.metadata[idxs[0][i]].get("intent", "general_feedback_other")
        } for i in range(top_k)]
        return precedents, top_sim, (top_sim >= config.CONFIDENCE_THRESHOLD)

    def retrieve_bm25(self, query: str, top_k: int = 1):
        tokens = query.lower().split()
        scores = self.bm25_index.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [{
            "bm25_score": float(scores[i]),
            "historical_query": self.metadata[i]["customer_text"],
            "uber_reply": self.metadata[i]["uber_reply"],
            "intent": self.metadata[i].get("intent", "general_feedback_other")
        } for i in top_idxs]

    def classify_intent(self, query: str):
        """Target System classifier: Dense embeddings + Logistic Regression."""
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        probs = self.classifier.predict_proba(q_vec)[0]
        best_idx = np.argmax(probs)
        return self.classifier.classes_[best_idx], float(probs[best_idx])

    def classify_intent_bm25(self, query: str):
        """Baseline 2 classifier: Nearest-neighbor intent via BM25 lexical match."""
        matches = self.retrieve_bm25(query, top_k=1)
        if matches:
            return matches[0]["intent"]
        return "general_feedback_other"
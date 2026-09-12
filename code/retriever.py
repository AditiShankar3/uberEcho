"""
retriever.py
Dual-retrieval engine (Dense FAISS + Sparse BM25) and calibrated intent classifier.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import config

INTENT_MAP = {
    "safety_critical": 0,
    "fare_billing_dispute": 1,
    "lost_item": 2,
    "driver_service_conduct": 3,
    "app_account_access": 4,
    "general_feedback_other": 5
}
REV_INTENT_MAP = {v: k for k, v in INTENT_MAP.items()}

class SupportEngine:
    def __init__(self, data_path: str = None, cache_dir: str = None):
        if data_path is None:
            data_path = os.path.join(config.DATA_DIR, "train_pairs.csv")
        if cache_dir is None:
            cache_dir = config.DATA_DIR

        self.data_path = data_path
        self.cache_dir = cache_dir

        self.retriever_cache_file = os.path.join(self.cache_dir, ".retriever_cache.pkl")
        self.index_file = os.path.join(self.cache_dir, ".faiss_index.bin")
        self.classifier_file = os.path.join(self.cache_dir, ".classifier_cache.pkl")

        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.metadata = []
        self.index = None
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_intents = []
        self.classifier = None

        if (os.path.exists(self.retriever_cache_file) and 
            os.path.exists(self.index_file) and 
            os.path.exists(self.classifier_file)):
            self._load_cache()
        else:
            self._build_and_cache()

    def _auto_label_text(self, text: str) -> str:
        t = str(text).lower()
        if re.search(r"\b(accident|crash|collision|police|assault|drunk|gun|knife|injured)\b", t): return "safety_critical"
        if re.search(r"\b(lost|left my|forgot|wallet|phone|keys|jacket|bag)\b", t): return "lost_item"
        if re.search(r"\b(cancellation fee|overcharge|double charge|surge price|refund|charged)\b", t): return "fare_billing_dispute"
        if re.search(r"\b(rude driver|attitude|dirty car|wrong route|reckless|cancelled on me)\b", t): return "driver_service_conduct"
        if re.search(r"\b(can't log in|login error|password reset|promo code|verification code)\b", t): return "app_account_access"
        return "general_feedback_other"

    def _build_and_cache(self):
        print(f"Building engine indices from {self.data_path} (Sample size: 3000)...", flush=True)
        df = pd.read_csv(self.data_path)
        sample_df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)

        queries = sample_df["customer_text"].astype(str).tolist()
        replies = sample_df["uber_reply"].astype(str).tolist()
        intents = [self._auto_label_text(q) for q in queries]

        # 1. Build Dense FAISS Index
        embeddings = self.encoder.encode(queries, show_progress_bar=False, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

        # 2. Build BM25 Index
        self.bm25_corpus = [q.lower().split() for q in queries]
        self.bm25_intents = intents
        self.bm25 = BM25Okapi(self.bm25_corpus)

        # 3. Build Metadata
        self.metadata = []
        for q, r, it in zip(queries, replies, intents):
            self.metadata.append({
                "historical_query": q,
                "uber_reply": r,
                "intent": it
            })

        # 4. Train Calibrated Classifier
        y = np.array([INTENT_MAP[it] for it in intents])
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.classifier.fit(embeddings, y)

        # Cache Artifacts
        os.makedirs(self.cache_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.retriever_cache_file, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "bm25_corpus": self.bm25_corpus,
                "bm25_intents": self.bm25_intents
            }, f)
        with open(self.classifier_file, "wb") as f:
            pickle.dump(self.classifier, f)
        print("Engine built and cached successfully!\n", flush=True)

    def _load_cache(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.retriever_cache_file, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.bm25_corpus = data["bm25_corpus"]
            self.bm25_intents = data["bm25_intents"]
            self.bm25 = BM25Okapi(self.bm25_corpus)
        with open(self.classifier_file, "rb") as f:
            self.classifier = pickle.load(f)

    def retrieve_dense(self, query: str, top_k: int = 2):
        emb = self.encoder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(emb, dtype=np.float32), top_k)
        top_sim = float(distances[0][0])
        precedents = [self.metadata[idx] for idx in indices[0]]
        is_confident = (top_sim >= config.CONFIDENCE_THRESHOLD)
        return precedents, top_sim, is_confident

    def retrieve_bm25(self, query: str, top_k: int = 1):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = int(np.argmax(scores))
        return [self.metadata[top_idx]]

    def classify_intent(self, query: str):
        emb = self.encoder.encode([query], normalize_embeddings=True)
        probs = self.classifier.predict_proba(emb)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return REV_INTENT_MAP[pred_idx], confidence

    def classify_intent_bm25(self, query: str) -> str:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = int(np.argmax(scores))
        return self.bm25_intents[top_idx]
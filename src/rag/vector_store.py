"""
Vector store for financial education articles using OpenAI embeddings and cosine similarity.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
from openai import OpenAI

from src.core.config import CONFIG, get_openai_api_key

EMBEDDING_MODEL = CONFIG.get("models", {}).get("embedding_model", "text-embedding-3-small")


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using OpenAI API."""
    if not texts:
        return []
    client = OpenAI(api_key=get_openai_api_key())
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


class FinancialVectorStore:
    """In-memory vector store with metadata filtering and relevance scoring."""

    def __init__(self):
        self.documents: list[dict] = []
        self.embeddings: list[list[float]] = []

    def load_kb_directory(self, kb_dir_path: Path) -> int:
        """Loads and indexes all JSON knowledge base files in a directory."""
        if not kb_dir_path.exists():
            return 0

        all_docs = []
        for json_file in kb_dir_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_docs.extend(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if not all_docs:
            return 0

        # Build embedding texts: Title + Category + Content
        texts_to_embed = [
            f"Category: {d.get('category', '')}\nTitle: {d.get('title', '')}\nContent: {d.get('content', '')}"
            for d in all_docs
        ]

        self.documents = all_docs
        self.embeddings = _embed_batch(texts_to_embed)
        return len(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None
    ) -> list[dict]:
        """Search articles with relevance scores and optional category filter."""
        if not self.documents:
            return []

        query_emb = _embed_batch([query])[0]

        scored_docs = []
        for doc, emb in zip(self.documents, self.embeddings):
            # Optional category filter
            if category and doc.get("category", "").lower() != category.lower():
                continue
            score = _cosine_similarity(query_emb, emb)
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                "id": doc.get("id", "N/A"),
                "category": doc.get("category", "General"),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "relevance_score": round(score, 4),
            })
        return results


# ── Global Singleton Loader ──────────────────────────────────────────
_finance_store: Optional[FinancialVectorStore] = None


def get_finance_vector_store() -> FinancialVectorStore:
    """Lazy-load singleton instance of the vector store."""
    global _finance_store
    if _finance_store is None:
        from src.core.config import BASE_DIR
        kb_path = BASE_DIR / CONFIG.get("rag", {}).get("kb_dir", "src/data/kb")
        _finance_store = FinancialVectorStore()
        total_indexed = _finance_store.load_kb_directory(kb_path)
        print(f"Indexed {total_indexed} financial articles into Vector Store.")
    return _finance_store
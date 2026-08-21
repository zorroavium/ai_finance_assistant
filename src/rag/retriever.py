"""
RAG Retrieval tools for financial literacy and tax education.
Provides confidence scoring and citation metadata.
"""

import json
import re
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.core.config import CONFIG
from src.rag.vector_store import get_finance_vector_store

CONFIDENCE_THRESHOLD = CONFIG.get("rag", {}).get("confidence_threshold", 0.35)
TOP_K = CONFIG.get("rag", {}).get("top_k", 3)

class FinancialSearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query regarding finance, investing, or taxes.")
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter: 'Investing Basics', 'Tax Accounts', 'Portfolio Management', 'Risk & Planning'."
    )

def validate_and_format_citations(text: str, retrieved_docs: list[dict]) -> str:
    """Ensure generated reference IDs exist in the retrieved document pool."""
    valid_ids = {doc["id"] for doc in retrieved_docs if "id" in doc}
    
    def _replace_tag(match):
        ref_id = match.group(1).strip()
        if ref_id in valid_ids:
            return f"[ref: {ref_id}]"
        return ""  # Strip hallucinations or nonexistent IDs
        
    return re.sub(r"\[ref:\s*([A-Za-z0-9_-]+)\]", _replace_tag, text)

@tool(args_schema=FinancialSearchInput)
def search_financial_kb(query: str, category: Optional[str] = None) -> str:
    """Search the curated financial knowledge base for educational content.

    Returns relevant articles with relevance scores and reference IDs.
    If the top score is below the threshold, confidence will be marked as 'low'.
    """
    store = get_finance_vector_store()
    results = store.search(query=query, top_k=TOP_K, category=category)

    if not results:
        return json.dumps({
            "status": "no_results",
            "message": "No matching financial articles found in knowledge base.",
            "confidence": "none",
            "results": []
        })

    top_score = results[0]["relevance_score"]
    confidence = "high" if top_score >= CONFIDENCE_THRESHOLD else "low"

    return json.dumps({
        "status": "results_found",
        "confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_score": top_score,
        "results": results,
    }, indent=2)
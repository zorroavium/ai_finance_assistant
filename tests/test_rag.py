import json
from src.rag.retriever import search_financial_kb, validate_and_format_citations


def test_citations_keep_only_retrieved_reference_ids():
    text = "Valid [ref: INV-001] and invalid [ref: FAKE-999]."
    docs = [{"id": "INV-001"}]

    assert validate_and_format_citations(text, docs) == "Valid [ref: INV-001] and invalid ."

def test_rag_retrieval_high_confidence():
    output = search_financial_kb.invoke({"query": "What is dollar cost averaging?"})
    data = json.loads(output)
    assert data["status"] == "results_found"
    assert data["confidence"] == "high"
    assert any("INV-002" in r["id"] or "DCA" in r["title"] for r in data["results"])

def test_rag_retrieval_tax_category():
    output = search_financial_kb.invoke({
        "query": "traditional vs roth ira rules",
        "category": "Tax Accounts"
    })
    data = json.loads(output)
    assert data["status"] == "results_found"
    assert data["results"][0]["category"] == "Tax Accounts"
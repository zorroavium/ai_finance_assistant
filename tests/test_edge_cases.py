"""
Edge cases, malformed payloads, rate-limit retries, and low-confidence RAG fallbacks.
"""

import json
import pytest
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.tools.goal_tools import project_goal_growth
from src.tools.market_tools import get_market_quote, get_market_history
from src.rag.retriever import search_financial_kb


def test_portfolio_malformed_json():
    res = json.loads(calculate_portfolio_metrics.invoke({"holdings_json": "INVALID_JSON"}))
    assert res.get("status") == "failed" or "error" in res


def test_portfolio_zero_total_value():
    holdings = [{"symbol": "CASH", "shares": 0, "price": 0.0, "category": "Cash", "expense_ratio": 0.0}]
    res = json.loads(calculate_portfolio_metrics.invoke({"holdings_json": json.dumps(holdings)}))
    assert "error" in res


def test_market_invalid_ticker_fallback_or_error():
    res = json.loads(get_market_quote.invoke({"ticker": "NONEXISTENT_TICKER_XYZ_123"}))
    assert "error" in res or res.get("status") == "failed"


def test_market_history_invalid_ticker():
    res = json.loads(get_market_history.invoke({"ticker": "NONEXISTENT_TICKER_XYZ_123", "period": "1mo"}))
    assert "error" in res or res.get("status") == "failed"


def test_goal_zero_return_calculation():
    res = json.loads(project_goal_growth.invoke({
        "initial_amount": 1000.0,
        "monthly_contribution": 100.0,
        "years": 1,
        "annual_return_pct": 0.0
    }))
    assert res["projected_median_value"] == 2200.0
    assert res["total_growth_interest"] == 0.0


def test_rag_low_confidence_query():
    # Out of domain query
    res = json.loads(search_financial_kb.invoke({"query": "quantum mechanical wavefunctions in superstring theory"}))
    assert res["status"] in ["results_found", "no_results"]
    if res["status"] == "results_found":
        assert res.get("confidence") in ["low", "high"]
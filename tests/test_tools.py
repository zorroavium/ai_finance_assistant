import json
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.tools.goal_tools import calculate_savings_plan, project_goal_growth
from src.core.config import set_runtime_api_keys


def test_calculate_portfolio_metrics_diversification_and_risk():
    holdings = [
        {"symbol": "VOO", "shares": 10, "price": 100.0, "category": "US Equities", "expense_ratio": 0.03},
        {"symbol": "BND", "shares": 10, "price": 100.0, "category": "Bonds", "expense_ratio": 0.05}
    ]
    res = json.loads(calculate_portfolio_metrics.invoke({
        "holdings_json": json.dumps(holdings),
        "user_risk_appetite": "Moderate"
    }))

    assert res["total_value"] == 2000.0
    assert res["allocation_percentages"]["US Equities"] == 50.0
    assert res["allocation_percentages"]["Bonds"] == 50.0
    assert "diversification_score" in res
    assert res["diversification_score"] > 0
    assert res["risk_alignment"] == "Aligned"


def test_calculate_portfolio_concentration_flag():
    holdings = [
        {"symbol": "AAPL", "shares": 90, "price": 100.0, "category": "US Equities", "expense_ratio": 0.0},
        {"symbol": "BND", "shares": 10, "price": 100.0, "category": "Bonds", "expense_ratio": 0.03}
    ]
    res = json.loads(calculate_portfolio_metrics.invoke({
        "holdings_json": json.dumps(holdings),
        "user_risk_appetite": "Conservative"
    }))
    assert len(res["concentrated_positions"]) > 0
    assert res["risk_alignment"] == "Over-exposed to Market Volatility"


def test_risk_aware_goal_growth():
    res = json.loads(project_goal_growth.invoke({
        "initial_amount": 5000.0,
        "monthly_contribution": 500.0,
        "years": 10,
        "risk_profile": "Aggressive"
    }))
    assert res["projected_median_value"] > res["total_contributed"]
    assert res["optimistic_band_value"] > res["projected_median_value"]
    assert res["pessimistic_band_value"] < res["projected_median_value"]
    assert "90% Equities" in res["recommended_asset_mix"]


def test_calculate_savings_plan_reaches_goal():
    result = calculate_savings_plan(
        goal_amount=100000.0,
        timeline_years=10,
        current_savings=10000.0,
        expected_return=0.07,
    )

    assert result["monthly_contribution_required"] > 0
    assert abs(result["final_balance"] - 100000.0) < 1.0
    assert result["interest_earned"] > 0


def test_runtime_api_key_updates(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    set_runtime_api_keys("test-openai", "test-tavily")
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-openai"
    assert __import__("os").environ["TAVILY_API_KEY"] == "test-tavily"
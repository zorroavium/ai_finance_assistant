import json
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.tools.goal_tools import project_goal_growth


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
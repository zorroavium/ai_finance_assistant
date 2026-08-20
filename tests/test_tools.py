import json
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.tools.goal_tools import project_goal_growth

def test_calculate_portfolio_metrics():
    holdings = [
        {"shares": 10, "price": 100.0, "category": "US Equities", "expense_ratio": 0.03},
        {"shares": 10, "price": 100.0, "category": "Bonds", "expense_ratio": 0.05}
    ]
    res = json.loads(calculate_portfolio_metrics.invoke({"holdings_json": json.dumps(holdings)}))
    assert res["total_value"] == 2000.0
    assert res["allocation_percentages"]["US Equities"] == 50.0
    assert res["allocation_percentages"]["Bonds"] == 50.0

def test_project_goal_growth():
    res = json.loads(project_goal_growth.invoke({
        "initial_amount": 1000.0,
        "monthly_contribution": 100.0,
        "annual_return_pct": 0.0,
        "years": 1
    }))
    assert res["projected_value"] == 2200.0
    assert res["total_contributed"] == 2200.0

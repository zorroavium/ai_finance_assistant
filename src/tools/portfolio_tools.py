"""Deterministic portfolio metrics & allocation calculations."""
import json
from langchain.tools import tool
from pydantic import BaseModel, Field

class PortfolioInput(BaseModel):
    holdings_json: str = Field(..., description="JSON string list of holdings with symbol, shares, price, category, expense_ratio.")

@tool(args_schema=PortfolioInput)
def calculate_portfolio_metrics(holdings_json: str) -> str:
    """Calculates total value, asset allocation percentages, and weighted expense ratio."""
    try:
        holdings = json.loads(holdings_json)
        total_val = sum(h["shares"] * h["price"] for h in holdings)
        if total_val == 0:
            return json.dumps({"error": "Total portfolio value is zero."})

        allocations = {}
        weighted_er = 0.0

        for h in holdings:
            val = h["shares"] * h["price"]
            cat = h.get("category", "Other")
            allocations[cat] = allocations.get(cat, 0.0) + val
            weighted_er += (val / total_val) * h.get("expense_ratio", 0.0)

        allocation_pcts = {k: round((v / total_val) * 100, 2) for k, v in allocations.items()}

        return json.dumps({
            "total_value": round(total_val, 2),
            "allocation_percentages": allocation_pcts,
            "weighted_expense_ratio_pct": round(weighted_er, 4),
            "status": "success"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to compute metrics: {str(e)}"})

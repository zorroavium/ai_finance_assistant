"""
Deterministic portfolio metrics, Herfindahl-Hirschman diversification scoring,
and risk appetite alignment engine.
"""

import json
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


class PortfolioInput(BaseModel):
    holdings_json: str = Field(
        ...,
        description="JSON string list of holdings with symbol, shares, price, category, expense_ratio."
    )
    user_risk_appetite: Optional[str] = Field(
        default="Moderate",
        description="User risk tolerance: 'Conservative', 'Moderate', or 'Aggressive'."
    )


@tool(args_schema=PortfolioInput)
def calculate_portfolio_metrics(holdings_json: str, user_risk_appetite: str = "Moderate") -> str:
    """Calculates total value, asset allocation %, weighted expense ratio,

    diversification score (0-100), concentration risks, and risk appetite alignment.
    """
    try:
        holdings = json.loads(holdings_json)
        total_val = sum(h["shares"] * h["price"] for h in holdings)
        if total_val == 0:
            return json.dumps({"error": "Total portfolio value is zero."})

        allocations = {}
        weighted_er = 0.0
        symbols_weights = {}

        for h in holdings:
            val = h["shares"] * h["price"]
            cat = h.get("category", "Other")
            allocations[cat] = allocations.get(cat, 0.0) + val
            weighted_er += (val / total_val) * h.get("expense_ratio", 0.0)
            sym = h.get("symbol", "UNKNOWN").upper()
            symbols_weights[sym] = symbols_weights.get(sym, 0.0) + (val / total_val)

        allocation_pcts = {k: round((v / total_val) * 100, 2) for k, v in allocations.items()}

        # 1. Diversification Score via Normalized Herfindahl-Hirschman Index (HHI)
        # HHI is sum of squared weights (0 to 1). Lower HHI = higher diversification.
        hhi = sum(w ** 2 for w in symbols_weights.values())
        raw_div_score = (1.0 - hhi) / (1.0 - 0.05) if hhi < 1.0 else 0.0
        diversification_score = max(5.0, min(98.0, round(raw_div_score * 100, 1)))

        if diversification_score >= 80:
            div_rating = "Excellent (Broadly Diversified)"
        elif diversification_score >= 60:
            div_rating = "Moderate (Balanced Distribution)"
        else:
            div_rating = "Low (High Concentration Risk)"

        # 2. Risk Appetite Assessment
        equity_pct = sum(
            v for k, v in allocation_pcts.items()
            if any(term in k.lower() for term in ["equity", "equities", "stock"])
        )
        bond_pct = sum(
            v for k, v in allocation_pcts.items()
            if any(term in k.lower() for term in ["bond", "fixed", "treasury"])
        )
        cash_pct = max(0.0, round(100.0 - equity_pct - bond_pct, 2))

        risk_targets = {
            "Conservative": {"max_equity": 35.0, "min_bond": 50.0},
            "Moderate": {"max_equity": 70.0, "min_bond": 25.0},
            "Aggressive": {"max_equity": 95.0, "min_bond": 0.0},
        }

        target = risk_targets.get(user_risk_appetite, risk_targets["Moderate"])
        risk_status = "Aligned"
        recommendation = f"Portfolio allocation matches your stated {user_risk_appetite} profile."

        if user_risk_appetite == "Conservative" and equity_pct > target["max_equity"]:
            risk_status = "Over-exposed to Market Volatility"
            recommendation = (
                f"Equity exposure ({equity_pct:.1f}%) exceeds the Conservative ceiling "
                f"({target['max_equity']}%). Consider rebalancing toward fixed income (e.g., BND, T-Bills)."
            )
        elif user_risk_appetite == "Aggressive" and (bond_pct + cash_pct) > 30.0:
            risk_status = "Under-exposed for Aggressive Growth"
            recommendation = (
                f"Defensive allocation ({bond_pct + cash_pct:.1f}%) may cause drag on long-term compound growth."
            )

        # 3. Concentration Risk Flag (>25% in a single position)
        concentrated = [
            f"{sym} ({w * 100:.1f}%)" for sym, w in symbols_weights.items() if w > 0.25
        ]

        return json.dumps({
            "total_value": round(total_val, 2),
            "allocation_percentages": allocation_pcts,
            "weighted_expense_ratio_pct": round(weighted_er, 4),
            "diversification_score": diversification_score,
            "diversification_rating": div_rating,
            "user_risk_appetite": user_risk_appetite,
            "risk_alignment": risk_status,
            "recommendation": recommendation,
            "concentrated_positions": concentrated,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to compute metrics: {str(e)}", "status": "failed"})
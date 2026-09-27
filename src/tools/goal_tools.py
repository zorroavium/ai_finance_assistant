"""
Risk-aware financial goal projections with compound interest and scenario bands.
"""

import json
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


class GoalProjectionInput(BaseModel):
    initial_amount: float = Field(default=0.0, description="Current lump sum invested.")
    monthly_contribution: float = Field(..., description="Monthly contribution amount.")
    years: int = Field(..., description="Investment horizon in years.")
    risk_profile: Optional[str] = Field(
        default="Moderate",
        description="Risk profile: 'Conservative', 'Moderate', or 'Aggressive'."
    )
    annual_return_pct: Optional[float] = Field(
        default=None,
        description="Optional custom annual rate of return override."
    )


def calculate_savings_plan(
    goal_amount: float,
    timeline_years: int,
    current_savings: float = 0.0,
    expected_return: float = 0.07,
) -> dict:
    """Calculate the monthly contribution required to reach a savings goal."""
    months = timeline_years * 12
    monthly_rate = expected_return / 12

    if months <= 0:
        raise ValueError("timeline_years must be greater than zero")

    current_future_value = current_savings * ((1 + monthly_rate) ** months)
    remaining = max(0.0, goal_amount - current_future_value)
    if monthly_rate == 0:
        monthly_contribution = remaining / months
    else:
        monthly_contribution = remaining * monthly_rate / (((1 + monthly_rate) ** months) - 1)

    balance = current_savings
    total_contributions = 0.0
    interest_earned = 0.0
    balances = [balance]
    for _ in range(months):
        interest = balance * monthly_rate
        balance += interest + monthly_contribution
        interest_earned += interest
        total_contributions += monthly_contribution
        balances.append(balance)

    return {
        "goal_amount": goal_amount,
        "timeline_years": timeline_years,
        "current_savings": current_savings,
        "monthly_contribution_required": round(monthly_contribution, 2),
        "total_contributions": round(total_contributions, 2),
        "interest_earned": round(interest_earned, 2),
        "final_balance": round(balance, 2),
        "surplus": round(balance - goal_amount, 2),
        "feasible": monthly_contribution <= 10000,
        "monthly_balances": balances,
    }
@tool(args_schema=GoalProjectionInput)
def project_goal_growth(
    initial_amount: float,
    monthly_contribution: float,
    years: int,
    risk_profile: str = "Moderate",
    annual_return_pct: Optional[float] = None
) -> str:
    """Projects future wealth with compound interest, pessimistic/optimistic return bands,
    and asset-mix suggestions based on risk profile.
    """
    profile_specs = {
        "Conservative": {
            "expected": 4.5,
            "low": 3.0,
            "high": 6.0,
            "asset_mix": "30% Equities / 70% Fixed Income & Cash"
        },
        "Moderate": {
            "expected": 7.5,
            "low": 5.0,
            "high": 10.0,
            "asset_mix": "60% Equities / 40% Fixed Income"
        },
        "Aggressive": {
            "expected": 10.0,
            "low": 6.5,
            "high": 13.5,
            "asset_mix": "90% Equities / 10% Fixed Income"
        }
    }

    spec = profile_specs.get(risk_profile, profile_specs["Moderate"])
    mean_rate = annual_return_pct if annual_return_pct is not None else spec["expected"]

    def _calc_future_value(rate_pct: float) -> float:
        r = (rate_pct / 100.0) / 12.0
        n = years * 12
        if r == 0:
            return initial_amount + (monthly_contribution * n)
        return (initial_amount * ((1 + r) ** n)) + (monthly_contribution * (((1 + r) ** n - 1) / r))

    projected_median = _calc_future_value(mean_rate)
    projected_low = _calc_future_value(spec["low"])
    projected_high = _calc_future_value(spec["high"])
    total_contributed = initial_amount + (monthly_contribution * years * 12)
    total_growth = projected_median - total_contributed

    return json.dumps({
        "projected_median_value": round(projected_median, 2),
        "projected_value": round(projected_median, 2),  # Compatibility alias
        "pessimistic_band_value": round(projected_low, 2),
        "optimistic_band_value": round(projected_high, 2),
        "total_contributed": round(total_contributed, 2),
        "total_growth_interest": round(total_growth, 2),
        "risk_profile": risk_profile,
        "assumed_annual_return": f"{mean_rate}%",
        "recommended_asset_mix": spec["asset_mix"],
        "years": years,
        "status": "success"
    }, indent=2)
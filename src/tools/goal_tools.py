"""Financial goal compound interest and timeline projections."""
import json
from langchain.tools import tool
from pydantic import BaseModel, Field

class GoalProjectionInput(BaseModel):
    initial_amount: float = Field(default=0.0, description="Current lump sum invested.")
    monthly_contribution: float = Field(..., description="Monthly contribution amount.")
    annual_return_pct: float = Field(default=7.0, description="Expected annual rate of return (e.g. 7.0 for 7%).")
    years: int = Field(..., description="Investment horizon in years.")

@tool(args_schema=GoalProjectionInput)
def project_goal_growth(initial_amount: float, monthly_contribution: float, annual_return_pct: float, years: int) -> str:
    """Projects future value of investments given regular contributions and expected annual return."""
    r = (annual_return_pct / 100.0) / 12.0
    n = years * 12

    if r == 0:
        fv = initial_amount + (monthly_contribution * n)
    else:
        fv = (initial_amount * ((1 + r) ** n)) + (monthly_contribution * (((1 + r) ** n - 1) / r))

    total_contributed = initial_amount + (monthly_contribution * n)
    total_interest = fv - total_contributed

    return json.dumps({
        "projected_value": round(fv, 2),
        "total_contributed": round(total_contributed, 2),
        "total_growth_interest": round(total_interest, 2),
        "years": years,
        "assumed_annual_return": f"{annual_return_pct}%"
    }, indent=2)

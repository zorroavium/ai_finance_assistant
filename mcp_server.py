from mcp.server.fastmcp import FastMCP
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.tools.goal_tools import project_goal_growth
from src.tools.market_tools import get_market_quote

mcp = FastMCP("FinanceAssistant")

@mcp.tool()
def get_quote(ticker: str) -> str:
    """Retrieve real-time market data."""
    return get_market_quote.invoke({"ticker": ticker})

@mcp.tool()
def evaluate_portfolio(holdings_json: str, user_risk_appetite: str = "Moderate") -> str:
    """Analyze portfolio allocation and risk metrics."""
    return calculate_portfolio_metrics.invoke({
        "holdings_json": holdings_json,
        "user_risk_appetite": user_risk_appetite
    })

if __name__ == "__main__":
    mcp.run()
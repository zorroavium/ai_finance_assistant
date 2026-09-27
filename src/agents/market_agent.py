"""
Market Intelligence Agent — Fetches real-time market data and contextualizes prices.
"""

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agents.base import contextualize_query
from src.core.config import CONFIG, get_openai_api_key
from src.tools.market_tools import get_market_quote, normalize_ticker
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")
TEMPERATURE = CONFIG.get("models", {}).get("temperature", 0.2)

MARKET_SYSTEM_PROMPT = """You are the Market Intelligence Specialist.
Your mission is to provide real-time ticker data, recent performance metrics, and contextual market analysis.

RULES:
1. Always call `get_market_quote` with the ticker symbol mentioned (e.g., SPY, VOO, AAPL).
2. Report the current price, previous close, percentage change, and 52-week range.
3. Compare the asset against broader market context when relevant.
4. If cached data or fallback data is used, note it transparently.
"""

_NON_TICKER_WORDS = {
    "A", "AN", "AND", "DATA", "FOR", "GET", "HOW", "I", "IS", "MARKET",
    "OF", "PORTION", "PRICE", "QUOTE", "REQUEST", "STOCK", "THE", "TODAY",
    "WHAT", "WITH",
}


def _extract_ticker(query: str) -> str | None:
    """Extract the likely ticker without relying on an LLM tool-call decision."""
    candidates = re.findall(r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9.\-]{0,9}(?![A-Za-z])", query)
    candidates = [candidate.upper() for candidate in candidates]
    candidates = [candidate for candidate in candidates if candidate not in _NON_TICKER_WORDS]
    return normalize_ticker(candidates[-1]) if candidates else None


def _format_market_result(query: str, quote: dict) -> str:
    """Format a tool response for the UI when no model synthesis is available."""
    if quote.get("status") != "success":
        suggestion = quote.get("suggested_ticker")
        if suggestion:
            return f"I could not find market data for `{quote.get('ticker')}`. Did you mean `{suggestion}`?"
        return quote.get("error", "Market data was unavailable for that symbol.")

    return (
        f"**{quote.get('name', quote.get('ticker'))} ({quote.get('ticker')})**\n\n"
        f"- Current price: ${quote.get('current_price', 0):,.2f}\n"
        f"- Previous close: ${quote.get('previous_close', 0):,.2f}\n"
        f"- Daily change: {quote.get('change_percent', 'N/A')}\n"
        f"- 52-week range: ${quote.get('fifty_two_week_low') or 0:,.2f} - "
        f"${quote.get('fifty_two_week_high') or 0:,.2f}\n"
        f"- Data status: {quote.get('freshness', 'unknown')}"
    )


def market_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "market_agent" or (isinstance(task, dict) and task.get("agent") == "market_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    ticker = _extract_ticker(query)
    if not ticker:
        content = "Please provide a stock or ETF ticker, such as AAPL, MSFT, SPY, or VOO."
        return {
            "market_messages": [HumanMessage(content=query)],
            "agent_results": [{"agent": "market_agent", "result": content}],
        }

    quote = json.loads(get_market_quote.invoke({"ticker": ticker}))
    tool_result = json.dumps(quote, indent=2)

    messages = [
        SystemMessage(content=MARKET_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(query, state)),
    ]
    messages.append(ToolMessage(content=tool_result, tool_call_id=f"market-{ticker}", name="get_market_quote"))
    content = _format_market_result(query, quote)

    # Use the model only to add context when configured; the quote and failure
    # handling above remain deterministic and available offline.
    if get_openai_api_key() and quote.get("status") == "success":
        try:
            llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=get_openai_api_key())
            response = llm.invoke([
                SystemMessage(content=MARKET_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {query}\nVerified market data:\n{tool_result}"),
            ])
            content = response.content
        except Exception:
            pass

    return {
        "market_messages": messages,
        "agent_results": [{"agent": "market_agent", "result": content}]
    }
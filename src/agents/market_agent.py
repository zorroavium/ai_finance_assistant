"""
Market Intelligence Agent — Fetches real-time market data and contextualizes prices.
"""

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.core.config import CONFIG, OPENAI_API_KEY
from src.tools.market_tools import get_market_quote
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


def market_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "market_agent" or (isinstance(task, dict) and task.get("agent") == "market_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
    tools = [get_market_quote]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=MARKET_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_res = get_market_quote.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_res),
                tool_call_id=tool_call["id"],
                name="get_market_quote"
            ))
        final_resp = llm.invoke(messages)
        content = final_resp.content
    else:
        content = response.content

    return {
        "market_messages": messages,
        "agent_results": [{"agent": "market_agent", "result": content}]
    }
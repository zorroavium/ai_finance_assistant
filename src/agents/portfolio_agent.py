"""
Portfolio Analysis Agent — Calculates allocation, diversification, and fee metrics.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agents.base import contextualize_query
from src.core.config import CONFIG, get_openai_api_key
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")
TEMPERATURE = CONFIG.get("models", {}).get("temperature", 0.2)

PORTFOLIO_SYSTEM_PROMPT = """You are the Portfolio Analysis Specialist.
Your mission is to evaluate investment portfolios and provide objective metrics.

RULES:
1. If the user provides holdings or asks to analyze a portfolio, invoke `calculate_portfolio_metrics`.
2. Break down the asset allocation (Equities vs Bonds vs Cash), total portfolio value, and weighted expense ratio.
3. Highlight concentration risks (e.g., if a single equity or category exceeds 20-30%).
4. Recommend diversification or rebalancing strategies where appropriate.
5. Remind users that this analysis is educational and not personalized investment advice.
"""


def portfolio_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "portfolio_agent" or (isinstance(task, dict) and task.get("agent") == "portfolio_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=get_openai_api_key())
    tools = [calculate_portfolio_metrics]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=PORTFOLIO_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(query, state))
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_res = calculate_portfolio_metrics.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_res),
                tool_call_id=tool_call["id"],
                name="calculate_portfolio_metrics"
            ))
        final_resp = llm.invoke(messages)
        content = final_resp.content
    else:
        content = response.content

    return {
        "portfolio_messages": messages,
        "agent_results": [{"agent": "portfolio_agent", "result": content}]
    }
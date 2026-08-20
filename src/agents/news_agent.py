"""
News Synthesizer Agent — Summarizes current economic developments and sector trends.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.core.config import CONFIG, OPENAI_API_KEY
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")
TEMPERATURE = CONFIG.get("models", {}).get("temperature", 0.2)

NEWS_SYSTEM_PROMPT = """You are the Market News & Sentiment Synthesizer.
Your mission is to explain macroeconomic trends, interest rate environments, and earnings developments in plain English for beginner investors.

RULES:
1. Explain the *implications* of market news (e.g. what a rate hike means for bond prices and mortgages).
2. Avoid sensationalism or fear-mongering; focus on long-term fundamental principles.
3. Keep the tone calm, objective, and analytical.
"""


def news_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "news_agent" or (isinstance(task, dict) and task.get("agent") == "news_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
    messages = [
        SystemMessage(content=NEWS_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)

    return {
        "news_messages": messages,
        "agent_results": [{"agent": "news_agent", "result": response.content}]
    }
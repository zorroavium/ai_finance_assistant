"""
General Chat Agent — Handles greetings, pleasantries, and out-of-scope interactions.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.base import contextualize_query
from src.core.config import CONFIG, get_openai_api_key
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")

CHAT_SYSTEM_PROMPT = """You are the friendly concierge for the AI Finance Assistant.
Handle greetings, goodbyes, and explain what topics you can help with (Investing Basics, Portfolio Analysis, Real-time Market Quotes, Goal Planning, Tax Account Rules, and Market News).
"""


def general_chat_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=0.7, api_key=get_openai_api_key())
    messages = [
        SystemMessage(content=CHAT_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(query, state))
    ]
    response = llm.invoke(messages)
    return {
        "agent_results": [{"agent": "general_chat", "result": response.content}]
    }
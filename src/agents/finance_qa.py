"""
Finance Q&A Agent — Answers general financial education questions grounded in the KB.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agents.base import contextualize_query
from src.core.config import CONFIG, get_openai_api_key
from src.rag.retriever import search_financial_kb, validate_and_format_citations
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")
TEMPERATURE = CONFIG.get("models", {}).get("temperature", 0.2)

QA_SYSTEM_PROMPT = """You are the Finance Q&A Specialist for the Finnie - Personal Finance Agent.
Your mission is to provide clear, educational, and jargon-free explanations of financial concepts.

RULES:
1. Always call `search_financial_kb` to retrieve verified educational content.
2. Base your explanation strictly on the retrieved articles.
3. Cite sources with their Reference ID (e.g., [ref: INV-001]) inline.
4. If confidence is low or no articles match, answer conservatively based on general financial principles and state that the specific article wasn't in the internal KB.
5. NEVER provide speculative stock picks or individual investment advice. Keep it educational.
"""


def finance_qa_node(state: FinanceAssistantState) -> dict:
    """Finance Q&A agent execution node with cyclical tool loop."""
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "finance_qa" or (isinstance(task, dict) and task.get("agent") == "finance_qa"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=get_openai_api_key())
    tools = [search_financial_kb]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(f"Explain the following financial concept: {query}", state))
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Tool Execution Loop
    if response.tool_calls:
        retrieved_docs = []
        for tool_call in response.tool_calls:
            tool_res = search_financial_kb.invoke(tool_call["args"])
            try:
                retrieved_docs.extend(json.loads(tool_res).get("results", []))
            except (TypeError, ValueError):
                pass
            messages.append(ToolMessage(
                content=str(tool_res),
                tool_call_id=tool_call["id"],
                name="search_financial_kb"
            ))
        final_resp = llm.invoke(messages)
        content = validate_and_format_citations(final_resp.content, retrieved_docs)
    else:
        content = response.content

    return {
        "qa_messages": messages,
        "agent_results": [{"agent": "finance_qa", "result": content}]
    }
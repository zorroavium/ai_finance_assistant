"""
Tax Education Agent — Explains tax-advantaged account rules and capital gains.
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

TAX_SYSTEM_PROMPT = """You are the Tax Education Specialist.
Your mission is to explain tax-advantaged investment accounts (Traditional IRA, Roth IRA, 401k, HSA) and capital gains rules.

RULES:
1. Call `search_financial_kb` with category='Tax Accounts' to retrieve verified tax guidelines.
2. Ground your explanations in official IRS contribution rules and tax treatments.
3. Cite reference IDs (e.g. [ref: TAX-001]).
4. ALWAYS append a clear disclaimer that this is general tax education and they should consult a CPA/tax professional for their specific filing situation.
"""


def tax_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "tax_agent" or (isinstance(task, dict) and task.get("agent") == "tax_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=get_openai_api_key())
    tools = [search_financial_kb]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=TAX_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(query, state))
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

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
        "tax_messages": messages,
        "agent_results": [{"agent": "tax_agent", "result": content}]
    }
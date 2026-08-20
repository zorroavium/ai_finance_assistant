"""
Tax Education Agent — Explains tax-advantaged account rules and capital gains.
"""

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.core.config import CONFIG, OPENAI_API_KEY
from src.rag.retriever import search_financial_kb
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

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
    tools = [search_financial_kb]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=TAX_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_res = search_financial_kb.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_res),
                tool_call_id=tool_call["id"],
                name="search_financial_kb"
            ))
        final_resp = llm.invoke(messages)
        content = final_resp.content
    else:
        content = response.content

    return {
        "tax_messages": messages,
        "agent_results": [{"agent": "tax_agent", "result": content}]
    }
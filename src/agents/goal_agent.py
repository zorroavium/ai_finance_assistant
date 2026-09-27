"""
Goal Planning Agent — Calculates compound growth projections and milestone timelines.
"""

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.agents.base import contextualize_query
from src.core.config import CONFIG, get_openai_api_key
from src.tools.goal_tools import project_goal_growth
from src.workflow.state import FinanceAssistantState

PRIMARY_MODEL = CONFIG.get("models", {}).get("primary_model", "gpt-4o")
TEMPERATURE = CONFIG.get("models", {}).get("temperature", 0.2)

GOAL_SYSTEM_PROMPT = """You are the Financial Goal Planning Specialist.
Your mission is to help users model retirement, home purchase, or wealth accumulation goals.

RULES:
1. Always call `project_goal_growth` to run deterministic future value calculations.
2. If parameters (initial deposit, monthly savings, years, return rate) are missing, infer standard baselines (e.g. 7% return for diversified index funds) and state your assumptions clearly.
3. Compare the total money contributed vs compound growth interest earned.
4. Structure timelines with actionable milestone checkpoints.
"""


def goal_agent_node(state: FinanceAssistantState) -> dict:
    query = state.get("user_query", "")
    for task in state.get("tasks", []):
        if getattr(task, "agent", "") == "goal_agent" or (isinstance(task, dict) and task.get("agent") == "goal_agent"):
            query = getattr(task, "query", query) if hasattr(task, "query") else task.get("query", query)
            break

    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=TEMPERATURE, api_key=get_openai_api_key())
    tools = [project_goal_growth]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=GOAL_SYSTEM_PROMPT),
        HumanMessage(content=contextualize_query(query, state))
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_res = project_goal_growth.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_res),
                tool_call_id=tool_call["id"],
                name="project_goal_growth"
            ))
        final_resp = llm.invoke(messages)
        content = final_resp.content
    else:
        content = response.content

    return {
        "goal_messages": messages,
        "agent_results": [{"agent": "goal_agent", "result": content}]
    }
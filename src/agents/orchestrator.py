"""
Orchestrator Agent — Classifies queries, breaks composite requests into tasks,
and routes tasks dynamically using LangGraph Send API.
"""

import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from src.core.config import CONFIG, OPENAI_API_KEY
from src.core.logger import log_routing
from src.workflow.state import FinanceAssistantState, RoutingDecision

ROUTER_MODEL = CONFIG.get("models", {}).get("router_model", "gpt-4o")

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Routing Orchestrator for the AI Finance Assistant.
Your job is to analyze user queries, decompose multi-intent questions into discrete tasks, and select the optimal specialist agent for each task.

SPECIALIST AGENTS AVAILABLE:
- `finance_qa`: General financial concepts, investing basics, dollar-cost averaging, ETFs vs index funds, bonds.
- `portfolio_agent`: Analyzing holdings, calculating asset allocations, expense ratios, concentration risks.
- `market_agent`: Real-time stock/ETF prices, moving averages, daily percentage changes, ticker data.
- `goal_agent`: Retirement calculators, compound interest projections, wealth milestones, savings timelines.
- `tax_agent`: Tax-advantaged accounts (Roth IRA, Traditional IRA, 401k), capital gains tax rules.
- `news_agent`: Macroeconomic commentary, market news sentiment, interest rate implications.
- `general_chat`: Greetings, small talk, pleasantries, off-topic requests.

ROUTING RULES:
1. Single intent: Assign to the exact matching specialist. Set `requires_synthesis = False`.
2. Multi-intent / composite questions: Decompose into individual tasks with isolated sub-queries. Set `requires_synthesis = True`.
   Example: "What is VOO price and how is it taxed in a Roth IRA?"
   -> Task 1: market_agent (query: "Get current price and data for VOO")
   -> Task 2: tax_agent (query: "Explain tax rules for holding ETFs in a Roth IRA")
3. Be precise with query extraction so sub-agents receive clean inputs.
"""


def classify_query(state: FinanceAssistantState) -> dict:
    """Classifies user intent and produces structured routing decision."""
    start_time = time.time()
    user_query = state.get("user_query", "")

    llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(RoutingDecision)

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    decision: RoutingDecision = structured_llm.invoke(messages)
    elapsed = f"{(time.time() - start_time) * 1000:.0f}ms"
    log_routing(decision.tasks, decision.requires_synthesis, elapsed)

    return {
        "tasks": decision.tasks,
        "requires_synthesis": decision.requires_synthesis
    }


def dispatch_to_agents(state: FinanceAssistantState):
    """Dynamic conditional edge using LangGraph Send API to fan out tasks."""
    tasks = state.get("tasks", [])
    if not tasks:
        return [Send("general_chat", state)]

    send_list = []
    for task in tasks:
        agent_name = getattr(task, "agent", "general_chat")
        send_list.append(Send(agent_name, state))
    return send_list
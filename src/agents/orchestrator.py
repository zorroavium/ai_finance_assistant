"""
Orchestrator Agent — Classifies queries, breaks composite requests into tasks,
and routes tasks dynamically using LangGraph Send API.
"""

import time

from langgraph.types import Send

from src.agents.router import RouterAgent
from src.core.config import CONFIG
from src.core.logger import log_routing
from src.workflow.state import AgentTask, FinanceAssistantState

ROUTER_MODEL = CONFIG.get("models", {}).get("router_model", "gpt-4o")

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Routing Orchestrator for the Finnie - Personal Finance Agent.
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
    """Classifies user intent and produces structured routing decision.

    The active app routes a single query to one or more specialist agents based on
    explicit keyword signals, which matches the reference multi-agent workflow.
    """
    start_time = time.time()
    user_query = state.get("user_query", "")
    query = user_query.lower()

    router = RouterAgent({
        "finance_qa": object(),
        "portfolio_agent": object(),
        "market_agent": object(),
        "goal_agent": object(),
        "tax_agent": object(),
        "news_agent": object(),
        "general_chat": object(),
    })

    detected_agents = []

    if any(keyword in query for keyword in ["portfolio", "allocation", "holdings", "diversification"]):
        detected_agents.append("portfolio_agent")
    if any(keyword in query for keyword in ["ticker", "stock", "etf", "market", "spy", "voo", "price", "trend", "nasdaq", "s&p", "index"]):
        detected_agents.append("market_agent")
    if any(keyword in query for keyword in ["retirement", "goal", "savings", "compound", "wealth", "milestone", "emergency fund", "401k plan"]):
        detected_agents.append("goal_agent")
    if any(keyword in query for keyword in ["tax", "roth", "traditional ira", "ira contribution", "capital gains", "hsa", "401k"]):
        detected_agents.append("tax_agent")
    if any(keyword in query for keyword in ["news", "interest rate", "fed", "inflation", "earnings", "market sentiment", "macro"]):
        detected_agents.append("news_agent")
    if any(keyword in query for keyword in ["hello", "hi", "thanks", "goodbye", "greeting"]):
        detected_agents.append("general_chat")

    if not detected_agents:
        detected_agents = [router._fallback_route({"query": user_query})]
    else:
        unique = []
        for agent_name in detected_agents:
            if agent_name not in unique:
                unique.append(agent_name)
        detected_agents = unique

    tasks = [
        AgentTask(
            agent=agent_name,
            query=_scoped_query(agent_name, user_query),
            focus=agent_name.replace("_agent", ""),
        )
        for agent_name in detected_agents
    ]

    requires_synthesis = len(tasks) > 1
    elapsed = f"{(time.time() - start_time) * 1000:.0f}ms"
    log_routing(tasks, requires_synthesis, elapsed)

    return {
        "tasks": tasks,
        "requires_synthesis": requires_synthesis,
    }


def _scoped_query(agent_name: str, query: str) -> str:
    """Give each specialist a focused request while preserving the original wording."""
    if agent_name == "market_agent":
        return f"Market data portion of the request: {query}"
    if agent_name == "tax_agent":
        return f"Tax education portion of the request: {query}"
    if agent_name == "portfolio_agent":
        return f"Portfolio analysis portion of the request: {query}"
    if agent_name == "goal_agent":
        return f"Financial goal portion of the request: {query}"
    if agent_name == "news_agent":
        return f"Current news portion of the request: {query}"
    return query


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
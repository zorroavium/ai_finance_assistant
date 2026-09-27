"""
LangGraph Workflow Assembly for the Finnie - Personal Finance Agent.
Connects orchestrator, specialist agent nodes, and synthesizer with in-memory checkpointer.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.workflow.state import FinanceAssistantState
from src.agents.orchestrator import classify_query, dispatch_to_agents
from src.agents.finance_qa import finance_qa_node
from src.agents.portfolio_agent import portfolio_agent_node
from src.agents.market_agent import market_agent_node
from src.agents.goal_agent import goal_agent_node
from src.agents.tax_agent import tax_agent_node
from src.agents.news_agent import news_agent_node
from src.agents.general_chat import general_chat_node
from src.agents.synthesizer import synthesizer_node


def build_finance_graph():
    """Builds and compiles the production StateGraph workflow."""
    workflow = StateGraph(FinanceAssistantState)

    # 1. Register Orchestrator & Synthesizer
    workflow.add_node("orchestrator", classify_query)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Register 6 Domain Agents + Chat
    workflow.add_node("finance_qa", finance_qa_node)
    workflow.add_node("portfolio_agent", portfolio_agent_node)
    workflow.add_node("market_agent", market_agent_node)
    workflow.add_node("goal_agent", goal_agent_node)
    workflow.add_node("tax_agent", tax_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("general_chat", general_chat_node)

    # 3. Connect Entry Point
    workflow.add_edge(START, "orchestrator")

    # 4. Connect Dynamic Fan-Out Conditional Edges
    workflow.add_conditional_edges(
        "orchestrator",
        dispatch_to_agents,
        [
            "finance_qa",
            "portfolio_agent",
            "market_agent",
            "goal_agent",
            "tax_agent",
            "news_agent",
            "general_chat",
        ],
    )

    # 5. Connect Fan-In Edges to Synthesizer
    workflow.add_edge("finance_qa", "synthesizer")
    workflow.add_edge("portfolio_agent", "synthesizer")
    workflow.add_edge("market_agent", "synthesizer")
    workflow.add_edge("goal_agent", "synthesizer")
    workflow.add_edge("tax_agent", "synthesizer")
    workflow.add_edge("news_agent", "synthesizer")
    workflow.add_edge("general_chat", "synthesizer")

    # 6. Synthesizer to END
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=InMemorySaver())
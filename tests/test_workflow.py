import uuid
from langchain_core.messages import HumanMessage
from src.workflow.graph import build_finance_graph


def test_single_agent_workflow():
    graph = build_finance_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    query = "What is the difference between index funds and active mutual funds?"
    result = graph.invoke(
        {
            "user_query": query,
            "messages": [HumanMessage(content=query)],
            "user_profile": {},
            "tasks": [],
            "requires_synthesis": False,
            "agent_results": None,
            "qa_messages": None,
            "portfolio_messages": None,
            "market_messages": None,
            "goal_messages": None,
            "news_messages": None,
            "tax_messages": None,
        },
        config=config,
    )

    assert "final_answer" in result
    assert len(result["final_answer"]) > 50
    assert "Disclaimer" in result["final_answer"]


def test_parallel_multi_agent_workflow():
    graph = build_finance_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    query = "What is the price of SPY and what are the Roth IRA contribution rules?"
    result = graph.invoke(
        {
            "user_query": query,
            "messages": [HumanMessage(content=query)],
            "user_profile": {},
            "tasks": [],
            "requires_synthesis": False,
            "agent_results": None,
            "qa_messages": None,
            "portfolio_messages": None,
            "market_messages": None,
            "goal_messages": None,
            "news_messages": None,
            "tax_messages": None,
        },
        config=config,
    )

    assert "final_answer" in result
    assert len(result["tasks"]) >= 2
    assert "Disclaimer" in result["final_answer"]
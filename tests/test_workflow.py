import uuid
from langchain_core.messages import HumanMessage
from src.workflow.graph import build_finance_graph
from src.agents.orchestrator import classify_query, dispatch_to_agents
from src.workflow.state import AgentTask


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


def test_orchestrator_scopes_multi_agent_tasks():
    result = classify_query({
        "user_query": "What is the price of SPY and what are the Roth IRA contribution rules?",
    })

    assert result["requires_synthesis"] is True
    assert {task.agent for task in result["tasks"]} == {"market_agent", "tax_agent"}
    assert all("portion of the request" in task.query for task in result["tasks"])


def test_orchestrator_routes_news_and_portfolio_intents():
    news_result = classify_query({"user_query": "Summarize recent Fed news and inflation trends."})
    portfolio_result = classify_query({"user_query": "Review my holdings and allocation."})

    assert {task.agent for task in news_result["tasks"]} == {"market_agent", "news_agent"}
    assert portfolio_result["tasks"][0].agent == "portfolio_agent"


def test_orchestrator_falls_back_for_general_finance_query():
    result = classify_query({"user_query": "Explain the basics of personal finance."})
    assert result["tasks"][0].agent == "finance_qa"


def test_dispatch_uses_general_chat_when_tasks_are_empty():
    sends = dispatch_to_agents({"tasks": []})
    assert len(sends) == 1
    assert sends[0].node == "general_chat"


def test_dispatch_fans_out_each_agent_task():
    sends = dispatch_to_agents({
        "tasks": [
            AgentTask(agent="market_agent", query="SPY"),
            AgentTask(agent="tax_agent", query="Roth IRA"),
        ]
    })
    assert [send.node for send in sends] == ["market_agent", "tax_agent"]
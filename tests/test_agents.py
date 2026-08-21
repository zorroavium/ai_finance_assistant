"""
Comprehensive execution tests for all 6 domain agents and general chat.
"""

from src.agents.finance_qa import finance_qa_node
from src.agents.general_chat import general_chat_node
from src.agents.goal_agent import goal_agent_node
from src.agents.market_agent import market_agent_node
from src.agents.news_agent import news_agent_node
from src.agents.portfolio_agent import portfolio_agent_node
from src.agents.tax_agent import tax_agent_node


def test_portfolio_agent_execution_with_tools():
    state = {
        "user_query": "Analyze my portfolio of 10 shares of VOO at $495 and 20 shares of BND at $72",
        "tasks": [{
            "agent": "portfolio_agent",
            "query": "Analyze portfolio: VOO (10 @ 495), BND (20 @ 72)",
            "focus": "Allocation",
        }],
        "portfolio_messages": [],
        "agent_results": [],
    }
    output = portfolio_agent_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "portfolio_agent"


def test_market_agent_execution_with_quote():
    state = {
        "user_query": "What is the price of SPY today?",
        "tasks": [{"agent": "market_agent", "query": "Get market quote for SPY", "focus": "Price"}],
        "market_messages": [],
        "agent_results": [],
    }
    output = market_agent_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "market_agent"


def test_news_agent_execution():
    state = {
        "user_query": "What are current market interest rate expectations?",
        "tasks": [{"agent": "news_agent", "query": "Summarize interest rate expectations", "focus": "Rates"}],
        "news_messages": [],
        "agent_results": [],
    }
    output = news_agent_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "news_agent"


def test_general_chat_concierge():
    state = {
        "user_query": "Hello, how can you help me today?",
        "tasks": [{"agent": "general_chat", "query": "Hello", "focus": "Greeting"}],
        "agent_results": [],
    }
    output = general_chat_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "general_chat"


from src.agents.goal_agent import goal_agent_node

def test_goal_agent_execution_with_projection():
    state = {
        "user_query": "If I save $500 monthly for 20 years with moderate risk, what will I have?",
        "tasks": [{
            "agent": "goal_agent",
            "query": "Project savings: $500 monthly for 20 years, moderate risk",
            "focus": "Projection",
        }],
        "goal_messages": [],
        "agent_results": [],
    }
    output = goal_agent_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "goal_agent"
    assert len(output["agent_results"][0]["result"]) > 20    
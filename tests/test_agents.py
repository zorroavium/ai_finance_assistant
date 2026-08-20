from src.agents.finance_qa import finance_qa_node
from src.agents.goal_agent import goal_agent_node
from src.agents.tax_agent import tax_agent_node

def test_finance_qa_node_execution():
    state = {
        "user_query": "What is an index fund?",
        "tasks": [{"agent": "finance_qa", "query": "What is an index fund?", "focus": "Index Funds"}],
        "qa_messages": [],
        "agent_results": []
    }
    output = finance_qa_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "finance_qa"
    assert len(output["agent_results"][0]["result"]) > 20

def test_goal_agent_node_execution():
    state = {
        "user_query": "If I save $500 monthly for 10 years at 7%, what will I have?",
        "tasks": [{"agent": "goal_agent", "query": "Save $500 monthly for 10 years at 7%", "focus": "Retirement"}],
        "goal_messages": [],
        "agent_results": []
    }
    output = goal_agent_node(state)
    assert len(output["agent_results"]) == 1
    assert output["agent_results"][0]["agent"] == "goal_agent"
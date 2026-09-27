from src.agents.router import RouterAgent


def test_router_routes_market_queries():
    router = RouterAgent({
        "finance_qa": object(),
        "market_agent": object(),
        "portfolio_agent": object(),
        "goal_agent": object(),
        "tax_agent": object(),
        "general_chat": object(),
    })

    route = router._fallback_route({"query": "What is the price of SPY today?"})
    assert route == "market_agent"


def test_router_routes_tax_queries():
    router = RouterAgent({
        "finance_qa": object(),
        "market_agent": object(),
        "portfolio_agent": object(),
        "goal_agent": object(),
        "tax_agent": object(),
        "general_chat": object(),
    })

    route = router._fallback_route({"query": "How does Roth IRA contribution work?"})
    assert route == "tax_agent"


def test_router_routes_greetings_to_general_chat():
    router = RouterAgent({
        "finance_qa": object(),
        "market_agent": object(),
        "portfolio_agent": object(),
        "goal_agent": object(),
        "tax_agent": object(),
        "general_chat": object(),
    })

    route = router._fallback_route({"query": "Hi there!"})
    assert route == "general_chat"


def test_router_falls_back_to_finance_qa_for_unknown_queries():
    router = RouterAgent({"finance_qa": object(), "general_chat": object()})
    assert router._fallback_route({"query": "What should I learn next?"}) == "finance_qa"


def test_router_uses_first_agent_when_finance_qa_is_unavailable():
    router = RouterAgent({"market_agent": object()})
    assert router._fallback_route({"query": "Unclear request"}) == "market_agent"


def test_router_handles_no_available_agents():
    import asyncio

    router = RouterAgent({})
    result = asyncio.run(router.process("Hello"))
    assert result.content == "general_chat"
    assert result.confidence == 0.0

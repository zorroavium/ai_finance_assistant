"""Router agent for keeping finance tasks targeted and consistent."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from src.agents.base import AgentResponse, BaseAgent


class RouterAgent(BaseAgent):
    """Select the best specialist for a finance query without over-routing."""

    def __init__(self, available_agents: Dict[str, BaseAgent]):
        self.available_agents = available_agents
        super().__init__(
            name="Router Agent",
            description="Routes finance questions to the most relevant specialist agent.",
            system_prompt=(
                "Route the user's question to the most relevant specialist agent. "
                "Use only the agent names supplied in the available_agents mapping."
            ),
        )

    def _get_topics(self):
        return ["routing", "intent_detection", "agent_selection"]

    def _process_response(
        self,
        response: AIMessage,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        selected_agent = str(response.content).strip()
        if selected_agent not in self.available_agents:
            selected_agent = self._fallback_route(context or {})
        return AgentResponse(
            content=selected_agent,
            confidence=0.9,
            suggestions=[],
            sources=["RouterAgent"],
            metadata={"selected_agent": selected_agent},
        )

    def _fallback_route(self, context: Dict[str, Any]) -> str:
        query = str(context.get("query") or context.get("user_query") or "").lower()

        if any(keyword in query for keyword in ["tax", "roth", "traditional ira", "ira contribution", "capital gains", "hsa", "401k"]):
            if "tax_agent" in self.available_agents:
                return "tax_agent"
        if "portfolio" in query or "allocation" in query or "holdings" in query:
            if "portfolio_agent" in self.available_agents:
                return "portfolio_agent"
        if any(keyword in query for keyword in ["ticker", "stock", "etf", "market", "spy", "voo", "price", "trend", "nasdaq", "s&p", "index"]):
            if "market_agent" in self.available_agents:
                return "market_agent"
        if any(keyword in query for keyword in ["retirement", "goal", "savings", "compound", "wealth", "milestone", "401k plan", "emergency fund"]):
            if "goal_agent" in self.available_agents:
                return "goal_agent"
        if any(keyword in query for keyword in ["hello", "hi", "thanks", "greeting", "goodbye"]):
            if "general_chat" in self.available_agents:
                return "general_chat"
        if "finance_qa" in self.available_agents:
            return "finance_qa"
        return next(iter(self.available_agents.keys()), "general_chat")

    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        if not self.available_agents:
            return AgentResponse(
                content="general_chat",
                confidence=0.0,
                suggestions=[],
                sources=["RouterAgent"],
                metadata={"reason": "No agents available"},
            )

        route = self._fallback_route({"query": query, **(context or {})})
        return AgentResponse(
            content=route,
            confidence=0.92,
            suggestions=[],
            sources=["RouterAgent"],
            metadata={"selected_agent": route},
        )

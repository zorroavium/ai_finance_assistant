"""Minimal shared agent contract for the finance assistant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import get_openai_api_key


class AgentResponse(BaseModel):
    """Structured response returned by all specialist agents."""

    content: str = Field(..., description="Main response text")
    confidence: float = Field(default=0.0, description="Confidence between 0 and 1")
    suggestions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _format_contextual_message(message: str, context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return message
    context_parts = [message]
    user_profile = context.get("user_profile") or {}
    if user_profile:
        experience = user_profile.get("experience_level", "beginner")
        goals = user_profile.get("goals", [])
        context_parts.append(f"User context: {experience} level investor.")
        if goals:
            context_parts.append(f"User goals: {', '.join(goals)}.")

    portfolio = context.get("portfolio") or {}
    holdings = portfolio.get("holdings", []) if isinstance(portfolio, dict) else []
    if holdings:
        symbols = ", ".join(str(item.get("symbol", "UNKNOWN")) for item in holdings[:10])
        context_parts.append(f"User portfolio holdings: {symbols}.")

    history = context.get("conversation_history", [])
    if history:
        context_parts.append("Recent conversation:\n" + "\n".join(history[-6:]))

    return "\n\n".join(context_parts)


def contextualize_query(query: str, state: Dict[str, Any]) -> str:
    """Build the prompt context shared by all graph-based specialist nodes."""
    context = state.get("context") or {}
    context.setdefault("user_profile", state.get("user_profile") or {})
    context.setdefault("portfolio", state.get("portfolio") or {})
    context.setdefault("conversation_history", state.get("conversation_history") or [])
    return _format_contextual_message(query, context)


class BaseAgent(ABC):
    """Shared base class for routing and specialist agents."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.llm = None

        if get_openai_api_key():
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,
                api_key=get_openai_api_key(),
            )

    def _default_system_prompt(self) -> str:
        return (
            f"You are {self.name}. "
            f"Your role is to help users with financial education. "
            f"Provide clear guidance and note when advice is educational only."
        )

    def _enhance_with_context(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return message
        return _format_contextual_message(message, context)

    def _call_llm(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        if self.llm is None:
            return (
                "The finance assistant is currently unavailable because no OpenAI API key is configured. "
                "Add OPENAI_API_KEY in the environment to enable live responses."
            )

        enhanced_message = self._enhance_with_context(user_message, context)
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=enhanced_message),
            ]
        )

        return str(getattr(response, "content", response))

    @abstractmethod
    def _process_response(
        self,
        response: AIMessage,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Convert the model answer into the standard structure."""

    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Default agent processing flow."""
        response_text = self._call_llm(query, context)
        fake_response = AIMessage(content=response_text)
        return self._process_response(fake_response, context=context)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "topics": getattr(self, "_get_topics", lambda: [])(),
        }

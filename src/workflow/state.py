"""Graph state and Pydantic schemas for the multi-agent finance workflow."""
import operator
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

def reset_or_add(left: Optional[list], right) -> list:
    if right is None:
        return []
    return (left or []) + (right or [])

class AgentTask(BaseModel):
    agent: Literal[
        "finance_qa",
        "portfolio_agent",
        "market_agent",
        "goal_agent",
        "news_agent",
        "tax_agent",
        "general_chat",
    ] = Field(description="Target specialist agent.")
    query: str = Field(description="Scoped sub-query for this specialist agent.")
    focus: str = Field(default="", description="2-4 word focus label.")

class RoutingDecision(BaseModel):
    tasks: list[AgentTask] = Field(description="1 or more agent tasks.")
    requires_synthesis: bool = Field(default=False, description="True if parallel outputs require merging.")

class FinanceAssistantState(TypedDict):
    user_query: str
    messages: Annotated[list[AnyMessage], operator.add]
    user_profile: dict
    tasks: list[AgentTask]
    requires_synthesis: bool
    agent_results: Annotated[list[dict], reset_or_add]
    final_answer: str

    # Agent scratchpads
    qa_messages: Annotated[list[AnyMessage], reset_or_add]
    portfolio_messages: Annotated[list[AnyMessage], reset_or_add]
    market_messages: Annotated[list[AnyMessage], reset_or_add]
    goal_messages: Annotated[list[AnyMessage], reset_or_add]
    news_messages: Annotated[list[AnyMessage], reset_or_add]
    tax_messages: Annotated[list[AnyMessage], reset_or_add]

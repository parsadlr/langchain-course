from typing import List

from pydantic import BaseModel, Field


class source(BaseModel):
    """Schema for the source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the response from the agent"""

    answer: str = Field(description="The answer to the question")
    sources: List[source] = Field(
        description="List of the sources used by the agent to answer the question",
        default_factory=list,
    )

"""
Base schema models for the Pydantic AI agent system.
These models define the core data structures used throughout the application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class Message(BaseModel):
    """Represents a message in a conversation."""
    role: str = Field(description="Role of the message sender (user, assistant, system)")
    content: str = Field(description="Content of the message")

class ToolCall(BaseModel):
    """Represents a tool call made by an agent."""
    tool_name: str = Field(description="Name of the tool being called")
    tool_input: Dict[str, Any] = Field(description="Input parameters for the tool")

class ToolResult(BaseModel):
    """Represents the result of a tool call."""
    tool_name: str = Field(description="Name of the tool that was called")
    result: Any = Field(description="Result returned by the tool")
    error: Optional[str] = Field(None, description="Error message if the tool call failed")

class AgentAction(BaseModel):
    """Represents an action taken by an agent."""
    action_type: str = Field(description="Type of action (message, tool_call)")
    content: Union[str, ToolCall] = Field(description="Content of the action")

class AgentThought(BaseModel):
    """Represents an agent's thought process."""
    thought: str = Field(description="The agent's thought process")
    
class AgentResponse(BaseModel):
    """Base class for agent responses."""
    thoughts: Optional[AgentThought] = Field(None, description="The agent's thought process")
    actions: List[AgentAction] = Field(default_factory=list, description="Actions taken by the agent")

"""
Response models for the Pydantic AI agent system.
These models define the structured responses returned by different agents.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from models.schemas.base import AgentResponse, AgentThought

class CodeGenerationResponse(AgentResponse):
    """Response model for code generation tasks."""
    code: str = Field(description="Generated code")
    language: str = Field(description="Programming language of the generated code")
    explanation: Optional[str] = Field(None, description="Explanation of the generated code")
    file_path: Optional[str] = Field(None, description="Suggested file path for the generated code")

class CodeAnalysisResponse(AgentResponse):
    """Response model for code analysis tasks."""
    analysis: str = Field(description="Analysis of the code")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Issues found in the code")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")

class SearchResponse(AgentResponse):
    """Response model for search tasks."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    summary: Optional[str] = Field(None, description="Summary of the search results")

class QuestionAnsweringResponse(AgentResponse):
    """Response model for question answering tasks."""
    answer: str = Field(description="Answer to the question")
    confidence: Optional[float] = Field(None, description="Confidence score for the answer")
    sources: Optional[List[str]] = Field(None, description="Sources used to generate the answer")

class PlanningResponse(AgentResponse):
    """Response model for planning tasks."""
    plan: List[str] = Field(description="Steps in the plan")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete the plan")
    prerequisites: Optional[List[str]] = Field(None, description="Prerequisites for the plan")

class ErrorResponse(AgentResponse):
    """Response model for error cases."""
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions to resolve the error")

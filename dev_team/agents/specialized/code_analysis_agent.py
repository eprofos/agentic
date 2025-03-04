"""
Code analysis agent for the Pydantic AI agent system.
This agent specializes in analyzing code to identify issues and suggest improvements.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai import ModelRetry

from agents.core.base_agent import BaseAgent
from models.responses.agent_responses import CodeAnalysisResponse
from utils.logging.logger import setup_logger
from tools.code_tools import (
    analyze_code_complexity,
    check_security_vulnerabilities,
    check_performance_issues,
    check_code_style
)

# Create a logger for this module
logger = setup_logger("code_analysis_agent")

@dataclass
class CodeAnalysisDependencies:
    """Dependencies for the code analysis agent."""
    language: str
    code: str
    framework: Optional[str] = None
    linting_rules: Optional[Dict[str, Any]] = None
    context: Optional[str] = None

class CodeAnalysisAgent(BaseAgent[CodeAnalysisDependencies, CodeAnalysisResponse]):
    """Agent specialized in analyzing code to identify issues and suggest improvements."""
    
    def __init__(self):
        """Initialize the code analysis agent with appropriate system prompt and tools."""
        system_prompt = """
        You are an expert code analysis assistant. Your task is to analyze code to identify issues,
        suggest improvements, and provide insights. Follow these guidelines:
        
        1. Identify potential bugs, security vulnerabilities, and performance issues
        2. Suggest improvements for code readability and maintainability
        3. Check for adherence to best practices and coding standards
        4. Consider edge cases and error handling
        5. Provide clear explanations for each issue and suggestion
        
        Use the provided tools to enhance your code analysis capabilities.
        """
        
        # Initialize the base agent first
        super().__init__(
            system_prompt=system_prompt,
            result_type=CodeAnalysisResponse,
            deps_type=CodeAnalysisDependencies,
            usage_limits=UsageLimits(
                request_limit=5,
                total_tokens_limit=8000
            ),
            model_settings={"temperature": 0.1},
            retries=2
        )
        
        # Now register tools and system prompts
        self._register_tools()
        self._register_system_prompts()
        
        logger.info("CodeAnalysisAgent initialized")
    
    def _register_tools(self):
        """Register tools for the code analysis agent."""
        self.add_tool(self._check_security_vulnerabilities, retries=2)
        self.add_tool(self._check_performance_issues, retries=1)
        self.add_tool(self._check_code_style, retries=1)
        self.add_tool(self._analyze_code_complexity, takes_ctx=False)
        
        logger.info("Registered tools for CodeAnalysisAgent")
    
    def _register_system_prompts(self):
        """Register dynamic system prompts for the code analysis agent."""
        self.add_system_prompt(self.add_language_framework_prompt)
        
        logger.info("Registered system prompts for CodeAnalysisAgent")
    
    @staticmethod
    def add_language_framework_prompt(ctx: RunContext[CodeAnalysisDependencies]) -> str:
        """
        Add language and framework information to the system prompt.
        
        Args:
            ctx: Run context with dependencies
            
        Returns:
            System prompt with language and framework information
        """
        language = ctx.deps.language
        framework = ctx.deps.framework
        
        prompt = f"You are analyzing code written in {language}."
        
        if framework:
            prompt += f" The code uses the {framework} framework."
        
        if ctx.deps.linting_rules:
            prompt += f" Apply the provided linting rules during analysis."
        
        return prompt
    
    @staticmethod
    async def _check_security_vulnerabilities(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities in the code."""
        return check_security_vulnerabilities(ctx.deps.code, ctx.deps.language)
    
    @staticmethod
    async def _check_performance_issues(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """Check for performance issues in the code."""
        return check_performance_issues(ctx.deps.code, ctx.deps.language)
    
    @staticmethod
    def _analyze_code_complexity(
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Analyze code complexity."""
        return analyze_code_complexity(code, language)
    
    @staticmethod
    async def _check_code_style(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """Check for code style issues."""
        return check_code_style(ctx.deps.code, ctx.deps.language)

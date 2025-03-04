"""
Base agent implementation for the Pydantic AI agent system.
This module provides the foundation for all agent implementations.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, AsyncIterable
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartStartEvent

from utils.config.config import config
from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("base_agent")

# Generic type for dependencies
T = TypeVar('T')
# Generic type for result
R = TypeVar('R')

class BaseAgent(Generic[T, R]):
    """
    Base agent class that wraps Pydantic AI's Agent class.
    
    Generic Parameters:
        T: Type of dependencies
        R: Type of result
    """
    
    def __init__(
        self,
        system_prompt: str,
        model_name: Optional[str] = None,
        result_type: Optional[Type[R]] = None,
        deps_type: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        usage_limits: Optional[UsageLimits] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        retries: int = 1
    ):
        """
        Initialize the base agent.
        
        Args:
            system_prompt: System prompt for the agent
            model_name: Name of the model to use (defaults to config)
            result_type: Type of result expected from the agent
            deps_type: Type of dependencies required by the agent
            tools: List of tools available to the agent
            usage_limits: Usage limits for the agent
            model_settings: Settings to fine-tune model behavior
            retries: Number of retries for model generation
        """
        # Use model from config if not provided
        if model_name is None:
            model_name = config.openrouter.model
        
        # Create OpenAI-compatible model for OpenRouter
        model = OpenAIModel(
            model_name,
            base_url=config.openrouter.api_base,
            api_key=config.openrouter.api_key
        )
        
        # Create Pydantic AI agent
        self.agent = PydanticAgent(
            model,
            system_prompt=system_prompt,
            result_type=result_type,
            deps_type=deps_type,
            tools=tools or [],
            model_settings=model_settings,
            retries=retries
        )
        
        # Store usage limits
        self.usage_limits = usage_limits
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
    
    async def run(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None) -> R:
        """
        Run the agent with the given prompt and dependencies.
        
        Args:
            prompt: User prompt to send to the agent
            deps: Dependencies required by the agent
            message_history: Previous messages for conversation context
            
        Returns:
            The result from the agent
        """
        logger.info(f"Running {self.__class__.__name__} with prompt: {prompt[:50]}...")
        
        # Create usage tracker
        usage = Usage()
        
        # Run the agent
        result = await self.agent.run(
            prompt,
            deps=deps,
            usage=usage,
            usage_limits=self.usage_limits,
            message_history=message_history
        )
        
        logger.info(f"Agent usage: {usage}")
        return result.data
    
    def run_sync(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None) -> R:
        """
        Run the agent synchronously with the given prompt and dependencies.
        
        Args:
            prompt: User prompt to send to the agent
            deps: Dependencies required by the agent
            message_history: Previous messages for conversation context
            
        Returns:
            The result from the agent
        """
        logger.info(f"Running {self.__class__.__name__} synchronously with prompt: {prompt[:50]}...")
        
        # Create usage tracker
        usage = Usage()
        
        # Run the agent
        result = self.agent.run_sync(
            prompt,
            deps=deps,
            usage=usage,
            usage_limits=self.usage_limits,
            message_history=message_history
        )
        
        logger.info(f"Agent usage: {usage}")
        return result.data
    
    async def run_stream(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None):
        """
        Run the agent with streaming response.
        
        Args:
            prompt: User prompt to send to the agent
            deps: Dependencies required by the agent
            message_history: Previous messages for conversation context
            
        Returns:
            Streamed response from the agent
        """
        logger.info(f"Streaming {self.__class__.__name__} with prompt: {prompt[:50]}...")
        
        # Create usage tracker
        usage = Usage()
        
        # Run the agent with streaming
        return await self.agent.run_stream(
            prompt,
            deps=deps,
            usage=usage,
            usage_limits=self.usage_limits,
            message_history=message_history
        )
    
    async def iter_run(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None):
        """
        Run the agent with node-by-node iteration.
        
        Args:
            prompt: User prompt to send to the agent
            deps: Dependencies required by the agent
            message_history: Previous messages for conversation context
            
        Returns:
            Agent run for iteration
        """
        logger.info(f"Iterating {self.__class__.__name__} with prompt: {prompt[:50]}...")
        
        # Create usage tracker
        usage = Usage()
        
        # Return agent run for iteration
        return self.agent.iter(
            prompt,
            deps=deps,
            usage=usage,
            usage_limits=self.usage_limits,
            message_history=message_history
        )
    
    def add_tool(self, tool_func: Any, takes_ctx: bool = True, retries: Optional[int] = None) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool_func: Tool function to add
            takes_ctx: Whether the tool function takes a RunContext parameter
            retries: Number of retries for this specific tool
        """
        # In pydantic_ai 0.0.31, we need to use the decorator pattern or pass tools during initialization
        # Since we can't use decorators after initialization, we'll need to create a new agent with the tools
        current_tools = getattr(self.agent, 'tools', [])
        
        # Create a new list of tools including the new one
        tools = list(current_tools)
        tools.append(tool_func)
        
        # Instead of recreating the agent, we'll just store the tools for initialization
        # and pass them to the specialized agent constructors
        self._tools = tools
        
        logger.info(f"Added tool {tool_func.__name__} to {self.__class__.__name__}")
    
    def add_system_prompt(self, prompt_func: Any) -> None:
        """
        Add a dynamic system prompt to the agent.
        
        Args:
            prompt_func: Function that returns a system prompt string
        """
        # Store the system prompt function for later use
        if not hasattr(self, '_system_prompt_funcs'):
            self._system_prompt_funcs = []
        
        self._system_prompt_funcs.append(prompt_func)
        
        logger.info(f"Added system prompt function {prompt_func.__name__} to {self.__class__.__name__}")

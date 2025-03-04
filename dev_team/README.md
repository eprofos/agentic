# PydanticAI Agent System

This project implements an agent system using PydanticAI, a powerful framework for building AI agents with structured responses, function tools, and dynamic system prompts.

## Overview

The system consists of several specialized agents:

- **CodeGenerationAgent**: Generates code based on user requirements
- **CodeAnalysisAgent**: Analyzes code to identify issues and suggest improvements
- **QuestionAnsweringAgent**: Answers questions about code, programming concepts, and technical topics

Each agent uses PydanticAI's features to enhance its capabilities:
- Function tools for retrieving information and performing actions
- Dynamic system prompts for context-aware instructions
- Structured responses for consistent output
- Conversation history for maintaining context across interactions

## Agent Components

In PydanticAI, an agent is a container for:

1. **System prompt(s)**: Instructions for the LLM written by the developer
2. **Function tool(s)**: Functions that the LLM may call to get information
3. **Structured result type**: The datatype the LLM must return
4. **Dependency type constraint**: Dependencies used by system prompts, tools, and result validators
5. **LLM model**: The model used by the agent
6. **Model Settings**: Settings to fine-tune requests

Our implementation wraps PydanticAI's Agent class with a BaseAgent class that provides additional functionality:

```python
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
        # Implementation...
```

## Function Tools

Function tools in PydanticAI provide a mechanism for models to retrieve extra information to help them generate a response. They're useful when it is impractical or impossible to put all the context an agent might need into the system prompt, or when you want to make agents' behavior more deterministic or reliable.

### Tool Registration Methods

There are several ways to register tools with an agent:

1. **`@agent.tool` decorator**: For tools that need access to the agent context via `RunContext`
2. **`@agent.tool_plain` decorator**: For tools that do not need access to the agent context
3. **`tools` keyword argument**: When initializing an `Agent` you can pass either plain functions or instances of `Tool`

In our implementation, we've added a helper method in the `BaseAgent` class to simplify tool registration:

```python
def add_tool(self, tool_func: Any, takes_ctx: bool = True, retries: Optional[int] = None) -> None:
    """
    Add a tool to the agent.
    
    Args:
        tool_func: Tool function to add
        takes_ctx: Whether the tool function takes a RunContext parameter
        retries: Number of retries for this specific tool
    """
    if takes_ctx:
        self.agent.add_tool(tool_func, retries=retries)
    else:
        self.agent.add_tool_plain(tool_func, retries=retries)
    logger.info(f"Added tool {tool_func.__name__} to {self.__class__.__name__}")
```

### Tool Examples

#### Tool with Context

```python
@staticmethod
async def search_documentation(
    ctx: RunContext[QuestionAnsweringDependencies],
    query: str,
    language: Optional[str] = None,
    framework: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search documentation for the given query.
    
    Args:
        ctx: Run context with dependencies
        query: Search query
        language: Programming language to search documentation for
        framework: Framework to search documentation for
        
    Returns:
        List of search results
    """
    # Implementation...
```

#### Plain Tool (No Context)

```python
@staticmethod
def get_current_date() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### Tool Registration

Tools are registered in the `_register_tools` method of each agent:

```python
def _register_tools(self):
    """Register tools for the question answering agent."""
    self.add_tool(self.search_documentation, retries=2)
    self.add_tool(self.generate_code_example, retries=1)
    
    # Register a plain tool that doesn't need context
    self.add_tool(self.get_current_date, takes_ctx=False)
```

## Dynamic System Prompts

Dynamic system prompts allow agents to adapt their instructions based on context. They are registered using the `@agent.system_prompt` decorator or the `add_system_prompt` method:

```python
@staticmethod
def add_language_framework_prompt(ctx: RunContext[CodeGenerationDependencies]) -> str:
    """
    Add language and framework information to the system prompt.
    
    Args:
        ctx: Run context with dependencies
        
    Returns:
        System prompt with language and framework information
    """
    language = ctx.deps.language
    framework = ctx.deps.framework
    
    prompt = f"You are generating code in {language}."
    
    if framework:
        prompt += f" The code should use the {framework} framework."
    
    if ctx.deps.libraries:
        libraries = ", ".join(ctx.deps.libraries)
        prompt += f" The following libraries are available: {libraries}."
    
    return prompt
```

## Running Agents

There are four ways to run an agent:

1. **`agent.run()`**: Asynchronous method that returns a completed response
2. **`agent.run_sync()`**: Synchronous method that returns a completed response
3. **`agent.run_stream()`**: Asynchronous method that returns a streamed response
4. **`agent.iter_run()`**: Asynchronous method that returns an agent run for node-by-node iteration

Our implementation provides wrapper methods for all these approaches:

```python
async def run(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None) -> R:
    # Implementation...

def run_sync(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None) -> R:
    # Implementation...

async def run_stream(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None):
    # Implementation...

async def iter_run(self, prompt: str, deps: Optional[T] = None, message_history: Optional[List[Any]] = None):
    # Implementation...
```

## Conversation History

The system supports maintaining conversation history across multiple interactions. This is implemented using the `message_history` parameter and the `new_messages()` method:

```python
# Update conversation history
generate_history = result.get("new_messages", [])
```

## Model Settings and Usage Limits

The system supports configuring model settings and usage limits:

```python
super().__init__(
    system_prompt=system_prompt,
    result_type=CodeGenerationResponse,
    deps_type=CodeGenerationDependencies,
    usage_limits=UsageLimits(
        request_limit=5,
        total_tokens_limit=8000
    ),
    model_settings={"temperature": 0.1},
    retries=2
)
```

## Running the System

The system can be run in different modes:

- **Generate**: Generate code based on a prompt
- **Analyze**: Analyze code for issues and suggestions
- **Answer**: Answer questions about code and programming concepts

Example:

```bash
python main.py --mode generate --prompt "Create a simple web server in Python" --language python
```

The system also supports an interactive mode with conversation history:

```bash
python main.py
```

## Benefits of PydanticAI

- **Type Safety**: Everything is defined with proper type annotations, making it type-safe
- **Documentation**: Docstrings are used to provide descriptions in the schema
- **Structured Responses**: Agents return structured data that can be validated
- **Context Access**: Tools and system prompts can access agent context when needed
- **Flexibility**: Multiple ways to run agents (sync, async, streaming, iteration)
- **Conversation History**: Support for maintaining context across interactions
- **Model Settings**: Fine-tuning of model behavior with settings
- **Usage Limits**: Control of token usage and request limits

## Future Improvements

- Add more specialized agents for different tasks
- Implement RAG (Retrieval-Augmented Generation) using vector search
- Integrate with external APIs for more powerful tools
- Add support for streaming responses in the interactive mode
- Implement more sophisticated error handling and retry mechanisms

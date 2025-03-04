"""
Code generation agent for the Pydantic AI agent system.
This agent specializes in generating code based on user requirements.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from agents.core.base_agent import BaseAgent
from models.responses.agent_responses import CodeGenerationResponse
from utils.logging.logger import setup_logger
from tools.code_tools import (
    append_to_file,
    display_directory_structure,
    read_file,
    generate_code_skeleton
)

# Create a logger for this module
logger = setup_logger("code_generation_agent")

@dataclass
class CodeGenerationDependencies:
    """Dependencies for the code generation agent."""
    language: str
    framework: Optional[str] = None
    libraries: Optional[List[str]] = None
    context: Optional[str] = None

class CodeGenerationAgent(BaseAgent[CodeGenerationDependencies, CodeGenerationResponse]):
    """Agent specialized in generating code based on user requirements."""
    
    def __init__(self):
        """Initialize the code generation agent with appropriate system prompt and tools."""
        system_prompt = """
        You are an expert code generation assistant. Your task is to generate high-quality,
        well-documented code based on user requirements. Follow these guidelines:
        
        1. Write clean, efficient, and maintainable code
        2. Include appropriate comments and documentation
        3. Follow best practices for the specified language and framework
        4. Consider edge cases and error handling
        5. Provide explanations for complex or non-obvious parts
        
        Use the provided tools to enhance your code generation capabilities.
        """
        
        # Initialize the base agent first
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
        
        # Now register tools and system prompts
        self._register_tools()
        self._register_system_prompts()
        
        logger.info("CodeGenerationAgent initialized")
    
    def _register_tools(self):
        """Register tools for the code generation agent."""
        # Add tools using the BaseAgent wrapper methods
        self.add_tool(self.analyze_requirements, retries=2)
        self.add_tool(self.suggest_libraries, retries=1)
        self.add_tool(self._generate_code_skeleton, takes_ctx=False)
        self.add_tool(self._append_to_file, takes_ctx=False)
        self.add_tool(self._display_directory_structure, takes_ctx=False)
        self.add_tool(self._read_file, takes_ctx=False)
        
        logger.info("Registered tools for CodeGenerationAgent")
    
    def _register_system_prompts(self):
        """Register dynamic system prompts for the code generation agent."""
        self.add_system_prompt(self.add_language_framework_prompt)
        
        logger.info("Registered system prompts for CodeGenerationAgent")
    
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
    
    def _create_code_file(self, code: str, file_path: str) -> bool:
        """
        Create a file with the generated code.
        
        Args:
            code: Generated code content
            file_path: Path where to save the file
            
        Returns:
            bool: True if file was created successfully, False otherwise
        """
        logger.info(f"Creating code file at: {file_path}")
        
        # Create the file using append_to_file tool
        success = self._append_to_file(file_path, code)
        
        if success:
            logger.info(f"Successfully created file: {file_path}")
        else:
            logger.error(f"Failed to create file: {file_path}")
        
        return success
    
    @staticmethod
    async def analyze_requirements(
        ctx: RunContext[CodeGenerationDependencies],
        requirements: str
    ) -> Dict[str, Any]:
        """
        Analyze user requirements to extract key components and constraints.
        
        Args:
            ctx: Run context with dependencies
            requirements: User requirements for code generation
            
        Returns:
            Dictionary with analyzed requirements
        """
        logger.info(f"Analyzing requirements: {requirements[:50]}...")
        
        # In a real implementation, this might use another agent or more sophisticated logic
        # For now, we'll return a simple analysis
        components = ["component1", "component2"]
        constraints = ["constraint1", "constraint2"]
        
        return {
            "components": components,
            "constraints": constraints,
            "language": ctx.deps.language,
            "framework": ctx.deps.framework
        }
    
    @staticmethod
    async def suggest_libraries(
        ctx: RunContext[CodeGenerationDependencies],
        task_description: str
    ) -> List[Dict[str, str]]:
        """
        Suggest libraries for the given task and language.
        
        Args:
            ctx: Run context with dependencies
            task_description: Description of the task
            
        Returns:
            List of suggested libraries with descriptions
        """
        logger.info(f"Suggesting libraries for task: {task_description[:50]}...")
        
        # In a real implementation, this might use a database or API
        # For now, we'll return some example libraries based on language
        language = ctx.deps.language.lower()
        
        libraries = []
        if language == "python":
            libraries = [
                {"name": "requests", "description": "HTTP library for making API calls"},
                {"name": "pandas", "description": "Data analysis and manipulation library"}
            ]
        elif language == "javascript":
            libraries = [
                {"name": "axios", "description": "Promise-based HTTP client"},
                {"name": "lodash", "description": "Utility library for common operations"}
            ]
        
        return libraries
    
    def _get_default_file_path(self, language: str) -> str:
        """
        Get a default file path based on the programming language.
        
        Args:
            language: Programming language
            
        Returns:
            str: Default file path
        """
        # Map of language to file extension
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "html": ".html",
            "css": ".css"
        }
        
        # Get extension for language, default to .txt
        ext = extensions.get(language.lower(), ".txt")
        
        # Create a basic file name
        if language.lower() == "python" and ext == ".py":
            return "simple_server.py"  # Special case for Python web server
        
        return f"generated_code{ext}"
    
    async def run(self, prompt: str, deps: CodeGenerationDependencies, message_history: Optional[List[Any]] = None) -> CodeGenerationResponse:
        """
        Run the code generation agent.
        
        Args:
            prompt: User prompt describing the code to generate
            deps: Dependencies for code generation
            message_history: Previous messages for conversation context
            
        Returns:
            CodeGenerationResponse with generated code and metadata
        """
        # Call parent's run method to get the response
        response = await super().run(prompt, deps, message_history)
        
        # Get the file path from the response or use default
        file_path = response.file_path or self._get_default_file_path(response.language)
        
        # Create the file with generated code
        if self._create_code_file(response.code, file_path):
            # Update response with actual file path
            response.file_path = file_path
        
        return response
    
    @staticmethod
    def _generate_code_skeleton(
        type: str,
        name: str,
        language: str,
        params: Optional[List[str]] = None,
        return_type: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Wrapper for generate_code_skeleton tool."""
        return generate_code_skeleton(
            language=language,
            type=type,
            name=name,
            params=params,
            return_type=return_type,
            description=description
        )
    
    @staticmethod
    def _append_to_file(file_path: str, content: Union[str, bytes], binary: bool = False) -> bool:
        """Wrapper for append_to_file tool."""
        return append_to_file(file_path, content, binary)
    
    @staticmethod
    def _display_directory_structure(
        path: str,
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None,
        sort_by: str = "name"
    ) -> str:
        """Wrapper for display_directory_structure tool."""
        return display_directory_structure(path, max_depth, exclude_patterns, sort_by)
    
    @staticmethod
    def _read_file(file_path: str, binary: bool = False) -> Optional[Union[str, bytes]]:
        """Wrapper for read_file tool."""
        return read_file(file_path, binary)

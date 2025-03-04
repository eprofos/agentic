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
        self.add_tool(self.generate_code_skeleton, takes_ctx=False)
        
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
    
    @staticmethod
    def generate_code_skeleton(
        type: str,
        name: str,
        language: str,
        params: Optional[List[str]] = None,
        return_type: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Generate a code skeleton for a function or class.
        
        Args:
            type: Type of skeleton (function, class, etc.)
            name: Name of the function or class
            language: Programming language for the skeleton
            params: List of parameter names
            return_type: Return type of the function
            description: Description for the docstring
            
        Returns:
            Generated code skeleton
        """
        logger.info(f"Generating {type} skeleton for {name} in {language}")
        
        params = params or []
        
        if language.lower() == "python":
            if type.lower() == "function":
                # Generate Python function skeleton
                code = f"def {name}({', '.join(params)}):\n"
                
                # Add docstring if description is provided
                if description:
                    code += f'    """\n    {description}\n'
                    
                    # Add parameter descriptions
                    if params:
                        code += "\n    Args:\n"
                        for param in params:
                            code += f"        {param}: Description of {param}\n"
                    
                    # Add return description if return type is provided
                    if return_type:
                        code += "\n    Returns:\n"
                        code += f"        {return_type}: Description of return value\n"
                    
                    code += '    """\n'
                
                # Add function body
                code += "    # TODO: Implement function\n"
                code += "    pass\n"
                
                return code
            elif type.lower() == "class":
                # Generate Python class skeleton
                code = f"class {name}:\n"
                
                # Add docstring if description is provided
                if description:
                    code += f'    """\n    {description}\n    """\n\n'
                
                # Add constructor
                code += "    def __init__(self"
                if params:
                    code += ", " + ", ".join(params)
                code += "):\n"
                
                # Add constructor body
                if params:
                    code += "        # Initialize attributes\n"
                    for param in params:
                        code += f"        self.{param} = {param}\n"
                else:
                    code += "        # TODO: Initialize attributes\n"
                    code += "        pass\n"
                
                return code
        elif language.lower() in ["javascript", "typescript"]:
            if type.lower() == "function":
                # Generate JavaScript function skeleton
                if language.lower() == "typescript" and return_type:
                    code = f"function {name}({', '.join(params)}): {return_type} {{\n"
                else:
                    code = f"function {name}({', '.join(params)}) {{\n"
                
                # Add function body
                if description:
                    code += f"    // {description}\n"
                code += "    // TODO: Implement function\n"
                
                # Add return statement if return type is provided
                if return_type and return_type.lower() != "void":
                    if return_type.lower() == "boolean":
                        code += "    return false;\n"
                    elif return_type.lower() in ["number", "int", "float"]:
                        code += "    return 0;\n"
                    elif return_type.lower() == "string":
                        code += "    return '';\n"
                    elif return_type.lower().startswith("array") or return_type.endswith("[]"):
                        code += "    return [];\n"
                    elif return_type.lower() == "object":
                        code += "    return {};\n"
                    else:
                        code += "    return null;\n"
                
                code += "}\n"
                
                return code
            elif type.lower() == "class":
                # Generate JavaScript class skeleton
                code = f"class {name} {{\n"
                
                # Add constructor
                code += "    constructor("
                if params:
                    code += ", ".join(params)
                code += ") {\n"
                
                # Add constructor body
                if description:
                    code += f"        // {description}\n"
                
                if params:
                    code += "        // Initialize attributes\n"
                    for param in params:
                        code += f"        this.{param} = {param};\n"
                else:
                    code += "        // TODO: Initialize attributes\n"
                
                code += "    }\n"
                code += "}\n"
                
                return code
        
        # Default case if language or type is not supported
        logger.warning(f"Unsupported language or type: {language}, {type}")
        return f"// TODO: Generate {type} skeleton for {name} in {language}"

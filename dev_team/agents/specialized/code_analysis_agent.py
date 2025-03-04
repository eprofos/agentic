"""
Code analysis agent for the Pydantic AI agent system.
This agent specializes in analyzing code to identify issues and suggest improvements.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai import ModelRetry

from agents.core.base_agent import BaseAgent
from models.responses.agent_responses import CodeAnalysisResponse
from utils.logging.logger import setup_logger

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
        self.add_tool(self.check_security_vulnerabilities, retries=2)
        self.add_tool(self.check_performance_issues, retries=1)
        self.add_tool(self.check_code_style, retries=1)
        self.add_tool(self.analyze_code_complexity, takes_ctx=False)
        
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
    async def check_security_vulnerabilities(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """
        Check for security vulnerabilities in the code.
        
        Args:
            ctx: Run context with dependencies
            
        Returns:
            List of security vulnerabilities found
        """
        logger.info("Checking for security vulnerabilities...")
        
        # In a real implementation, this might use a security scanning tool
        # For now, we'll return some example vulnerabilities based on language
        language = ctx.deps.language.lower()
        code = ctx.deps.code
        
        vulnerabilities = []
        
        # Simple pattern matching for common vulnerabilities
        if language == "python":
            if "eval(" in code:
                vulnerabilities.append({
                    "type": "security",
                    "severity": "high",
                    "description": "Use of eval() can lead to code injection vulnerabilities",
                    "line": code.find("eval(")  # Simplified line number detection
                })
            if "subprocess.call(" in code and "shell=True" in code:
                vulnerabilities.append({
                    "type": "security",
                    "severity": "high",
                    "description": "Shell=True in subprocess calls can lead to command injection",
                    "line": code.find("subprocess.call(")
                })
        elif language == "javascript":
            if "eval(" in code:
                vulnerabilities.append({
                    "type": "security",
                    "severity": "high",
                    "description": "Use of eval() can lead to code injection vulnerabilities",
                    "line": code.find("eval(")
                })
            if "innerHTML" in code:
                vulnerabilities.append({
                    "type": "security",
                    "severity": "medium",
                    "description": "Use of innerHTML can lead to XSS vulnerabilities",
                    "line": code.find("innerHTML")
                })
        
        return vulnerabilities
    
    @staticmethod
    async def check_performance_issues(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """
        Check for performance issues in the code.
        
        Args:
            ctx: Run context with dependencies
            
        Returns:
            List of performance issues found
        """
        logger.info("Checking for performance issues...")
        
        # In a real implementation, this might use a performance profiling tool
        # For now, we'll return some example issues based on language
        language = ctx.deps.language.lower()
        code = ctx.deps.code
        
        issues = []
        
        # Simple pattern matching for common performance issues
        if language == "python":
            if "+=" in code and any(container in code for container in ["list", "[]"]):
                issues.append({
                    "type": "performance",
                    "severity": "medium",
                    "description": "Using += for list concatenation in loops can be inefficient",
                    "line": code.find("+=")
                })
            if "for" in code and ".append" in code:
                issues.append({
                    "type": "performance",
                    "severity": "low",
                    "description": "Consider using list comprehensions instead of for loops with append",
                    "line": code.find("for")
                })
        elif language == "javascript":
            if "for (" in code and "length" in code:
                issues.append({
                    "type": "performance",
                    "severity": "low",
                    "description": "Consider caching array length in for loops",
                    "line": code.find("for (")
                })
        
        return issues
    
    @staticmethod
    def analyze_code_complexity(
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Analyze code complexity.
        
        Args:
            code: Code to analyze
            language: Programming language of the code
            
        Returns:
            Dictionary with complexity metrics
        """
        logger.info(f"Analyzing code complexity for language: {language}")
        
        # Initialize complexity metrics
        metrics = {
            "lines_of_code": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "function_count": 0,
            "class_count": 0,
            "cyclomatic_complexity": 0
        }
        
        # Count lines of code, comment lines, and blank lines
        lines = code.splitlines()
        metrics["lines_of_code"] = len(lines)
        
        for line in lines:
            line = line.strip()
            if not line:
                metrics["blank_lines"] += 1
            elif language.lower() == "python" and (line.startswith("#") or line.startswith('"""') or line.startswith("'''")):
                metrics["comment_lines"] += 1
            elif language.lower() in ["javascript", "typescript"] and (line.startswith("//") or line.startswith("/*") or line.startswith("*")):
                metrics["comment_lines"] += 1
        
        # Simple estimation of function and class counts
        if language.lower() == "python":
            metrics["function_count"] = code.count("def ")
            metrics["class_count"] = code.count("class ")
            
            # Simplified cyclomatic complexity calculation
            metrics["cyclomatic_complexity"] = (
                code.count("if ") + 
                code.count("elif ") + 
                code.count("for ") + 
                code.count("while ") + 
                code.count(" and ") + 
                code.count(" or ")
            )
        elif language.lower() in ["javascript", "typescript"]:
            metrics["function_count"] = (
                code.count("function ") + 
                code.count(" => {")
            )
            metrics["class_count"] = code.count("class ")
            
            # Simplified cyclomatic complexity calculation
            metrics["cyclomatic_complexity"] = (
                code.count("if (") + 
                code.count("else if") + 
                code.count("for (") + 
                code.count("while (") + 
                code.count(" && ") + 
                code.count(" || ")
            )
        
        logger.info(f"Code complexity metrics: {metrics}")
        return metrics
    
    @staticmethod
    async def check_code_style(
        ctx: RunContext[CodeAnalysisDependencies]
    ) -> List[Dict[str, Any]]:
        """
        Check for code style issues.
        
        Args:
            ctx: Run context with dependencies
            
        Returns:
            List of code style issues found
        """
        logger.info("Checking for code style issues...")
        
        # In a real implementation, this might use a linting tool
        # For now, we'll return some example issues based on language
        language = ctx.deps.language.lower()
        code = ctx.deps.code
        
        issues = []
        
        # Simple pattern matching for common style issues
        if language == "python":
            if "def " in code and not "def " + " " in code:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "description": "Missing space after 'def' keyword",
                    "line": code.find("def ")
                })
            if "if" in code and "if(" in code:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "description": "Missing space after 'if' keyword",
                    "line": code.find("if(")
                })
        elif language == "javascript":
            if "function" in code and "function(" in code:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "description": "Missing space after 'function' keyword",
                    "line": code.find("function(")
                })
            if "if" in code and "if(" in code:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "description": "Missing space after 'if' keyword",
                    "line": code.find("if(")
                })
        
        return issues

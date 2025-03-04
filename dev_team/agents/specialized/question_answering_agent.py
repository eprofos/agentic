"""
Question answering agent for the Pydantic AI agent system.
This agent specializes in answering questions about code, programming concepts, and technical topics.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from agents.core.base_agent import BaseAgent
from models.responses.agent_responses import QuestionAnsweringResponse
from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("question_answering_agent")

@dataclass
class QuestionAnsweringDependencies:
    """Dependencies for the question answering agent."""
    context: Optional[str] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    code_snippets: Optional[List[str]] = None

class QuestionAnsweringAgent(BaseAgent[QuestionAnsweringDependencies, QuestionAnsweringResponse]):
    """Agent specialized in answering questions about code, programming concepts, and technical topics."""
    
    def __init__(self):
        """Initialize the question answering agent with appropriate system prompt and tools."""
        system_prompt = """
        You are an expert technical assistant specializing in answering questions about code,
        programming concepts, and technical topics. Your task is to provide accurate, clear,
        and helpful answers. Follow these guidelines:
        
        1. Provide accurate and factual information
        2. Explain complex concepts in a clear and understandable way
        3. Include code examples when relevant
        4. Cite sources when available
        5. Acknowledge when you don't know the answer or when information might be outdated
        
        Use the provided tools to enhance your question answering capabilities.
        """
        
        # Initialize the base agent first
        super().__init__(
            system_prompt=system_prompt,
            result_type=QuestionAnsweringResponse,
            deps_type=QuestionAnsweringDependencies,
            usage_limits=UsageLimits(
                request_limit=5,
                total_tokens_limit=8000
            ),
            model_settings={"temperature": 0.2},
            retries=2
        )
        
        # Now register tools and system prompts
        self._register_tools()
        self._register_system_prompts()
        
        logger.info("QuestionAnsweringAgent initialized")
    
    def _register_tools(self):
        """Register tools for the question answering agent."""
        self.add_tool(self.search_documentation, retries=2)
        self.add_tool(self.generate_code_example, retries=1)
        
        # Register a plain tool that doesn't need context
        self.add_tool(self.get_current_date, takes_ctx=False)
        
        logger.info("Registered tools for QuestionAnsweringAgent")
    
    def _register_system_prompts(self):
        """Register dynamic system prompts for the question answering agent."""
        self.add_system_prompt(self.add_current_date_prompt)
        
        logger.info("Registered system prompts for QuestionAnsweringAgent")
    
    @staticmethod
    def add_current_date_prompt(ctx: RunContext[QuestionAnsweringDependencies]) -> str:
        """
        Add the current date to the system prompt.
        
        Args:
            ctx: Run context with dependencies
            
        Returns:
            System prompt with current date
        """
        return f"The current date is {datetime.now().strftime('%Y-%m-%d')}."
    
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
        logger.info(f"Searching documentation for: {query}")
        
        # In a real implementation, this might use a documentation search API
        # For now, we'll return some example results
        results = []
        
        if language and language.lower() == "python":
            results.append({
                "title": "Python Documentation",
                "url": "https://docs.python.org/3/",
                "snippet": "The Python documentation provides tutorials, library references, and language specifications."
            })
            
            if framework and framework.lower() == "django":
                results.append({
                    "title": "Django Documentation",
                    "url": "https://docs.djangoproject.com/",
                    "snippet": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design."
                })
        elif language and language.lower() == "javascript":
            results.append({
                "title": "JavaScript Documentation",
                "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
                "snippet": "JavaScript (JS) is a lightweight, interpreted, or just-in-time compiled programming language with first-class functions."
            })
            
            if framework and framework.lower() == "react":
                results.append({
                    "title": "React Documentation",
                    "url": "https://reactjs.org/docs/getting-started.html",
                    "snippet": "React is a JavaScript library for building user interfaces."
                })
        
        # Add a generic result if no specific results were found
        if not results:
            results.append({
                "title": "Programming Documentation",
                "url": "https://devdocs.io/",
                "snippet": "DevDocs combines multiple API documentations in a fast, organized, and searchable interface."
            })
        
        return results
    
    @staticmethod
    async def generate_code_example(
        ctx: RunContext[QuestionAnsweringDependencies],
        concept: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Generate a code example for the given concept and language.
        
        Args:
            ctx: Run context with dependencies
            concept: Programming concept to generate an example for
            language: Programming language for the example
            
        Returns:
            Dictionary with the code example and explanation
        """
        logger.info(f"Generating code example for concept: {concept} in language: {language}")
        
        # In a real implementation, this might use another agent or a code generation API
        # For now, we'll return some example code based on the concept and language
        language = language.lower()
        concept = concept.lower()
        
        code = ""
        explanation = ""
        
        if language == "python":
            if "list comprehension" in concept:
                code = """
# List comprehension example
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)  # Output: [1, 4, 9, 16, 25]
"""
                explanation = "List comprehensions provide a concise way to create lists based on existing lists. The example above creates a new list with the squares of each number in the original list."
            elif "class" in concept:
                code = """
# Class example
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."

# Create an instance of the Person class
person = Person("Alice", 30)
print(person.greet())  # Output: Hello, my name is Alice and I am 30 years old.
"""
                explanation = "Classes are used to create objects that bundle data and functionality together. The example above defines a Person class with name and age attributes, and a greet method."
        elif language == "javascript":
            if "arrow function" in concept:
                code = """
// Arrow function example
const numbers = [1, 2, 3, 4, 5];
const squared = numbers.map(x => x * x);
console.log(squared);  // Output: [1, 4, 9, 16, 25]
"""
                explanation = "Arrow functions provide a concise syntax for writing functions in JavaScript. The example above uses an arrow function with the map method to create a new array with the squares of each number in the original array."
            elif "class" in concept:
                code = """
// Class example
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    greet() {
        return `Hello, my name is ${this.name} and I am ${this.age} years old.`;
    }
}

// Create an instance of the Person class
const person = new Person("Alice", 30);
console.log(person.greet());  // Output: Hello, my name is Alice and I am 30 years old.
"""
                explanation = "Classes in JavaScript are primarily syntactical sugar over JavaScript's existing prototype-based inheritance. The example above defines a Person class with name and age properties, and a greet method."
        
        # If no specific example was found, provide a generic example
        if not code:
            if language == "python":
                code = """
# Hello World example
print("Hello, World!")
"""
                explanation = "This is a simple 'Hello, World!' program in Python, which is often the first program written when learning a new language."
            elif language == "javascript":
                code = """
// Hello World example
console.log("Hello, World!");
"""
                explanation = "This is a simple 'Hello, World!' program in JavaScript, which is often the first program written when learning a new language."
        
        return {
            "code": code.strip(),
            "explanation": explanation,
            "language": language
        }
    
    @staticmethod
    def get_current_date() -> str:
        """
        Get the current date and time.
        
        Returns:
            Current date and time as a string
        """
        from datetime import datetime
        logger.info("Getting current date")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

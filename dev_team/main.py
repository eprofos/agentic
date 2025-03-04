"""
Main application file for the Pydantic AI agent system.
This file serves as the entry point for the system.
"""

import asyncio
import argparse
from typing import Optional, Dict, Any, List, Union

from agents.specialized.code_generation_agent import CodeGenerationAgent, CodeGenerationDependencies
from agents.specialized.code_analysis_agent import CodeAnalysisAgent, CodeAnalysisDependencies
from agents.specialized.question_answering_agent import QuestionAnsweringAgent, QuestionAnsweringDependencies

from utils.logging.logger import setup_logger
from utils.config.config import config

# Create a logger for this module
logger = setup_logger("main")

class PydanticAIAgentSystem:
    """Main class for the Pydantic AI agent system."""
    
    def __init__(self):
        """Initialize the agent system."""
        logger.info("Initializing Pydantic AI agent system")
        
        # Initialize agents
        self.code_generation_agent = CodeGenerationAgent()
        self.code_analysis_agent = CodeAnalysisAgent()
        self.question_answering_agent = QuestionAnsweringAgent()
        
        logger.info("Pydantic AI agent system initialized")
    
    async def generate_code(
        self,
        prompt: str,
        language: str,
        framework: Optional[str] = None,
        libraries: Optional[List[str]] = None,
        context: Optional[str] = None,
        message_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate code based on the given prompt.
        
        Args:
            prompt: User prompt describing the code to generate
            language: Programming language to use
            framework: Framework to use (optional)
            libraries: List of libraries to use (optional)
            context: Additional context for code generation (optional)
            message_history: Previous messages for conversation context (optional)
            
        Returns:
            Generated code and related information
        """
        logger.info(f"Generating code for prompt: {prompt[:50]}...")
        
        # Create dependencies
        deps = CodeGenerationDependencies(
            language=language,
            framework=framework,
            libraries=libraries,
            context=context
        )
        
        # Run the code generation agent
        result = await self.code_generation_agent.run(prompt, deps, message_history)
        
        logger.info(f"Code generation completed for prompt: {prompt[:50]}")
        return {
            "code": result.code,
            "language": result.language,
            "explanation": result.explanation,
            "file_path": result.file_path,
            "new_messages": result.new_messages() if hasattr(result, 'new_messages') else []
        }
    
    async def analyze_code(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None,
        linting_rules: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        message_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze code and provide feedback.
        
        Args:
            code: Code to analyze
            language: Programming language of the code
            framework: Framework used in the code (optional)
            linting_rules: Custom linting rules (optional)
            context: Additional context for code analysis (optional)
            message_history: Previous messages for conversation context (optional)
            
        Returns:
            Analysis results and suggestions
        """
        logger.info(f"Analyzing code: {code[:50]}...")
        
        # Create dependencies
        deps = CodeAnalysisDependencies(
            language=language,
            code=code,
            framework=framework,
            linting_rules=linting_rules,
            context=context
        )
        
        # Run the code analysis agent
        result = await self.code_analysis_agent.run(code, deps, message_history)
        
        logger.info("Code analysis completed")
        return {
            "analysis": result.analysis,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "new_messages": result.new_messages() if hasattr(result, 'new_messages') else []
        }
    
    async def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict[str, Any]]] = None,
        code_snippets: Optional[List[str]] = None,
        message_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about code or programming concepts.
        
        Args:
            question: Question to answer
            context: Additional context for the question (optional)
            search_results: Pre-fetched search results (optional)
            code_snippets: Code snippets related to the question (optional)
            message_history: Previous messages for conversation context (optional)
            
        Returns:
            Answer and related information
        """
        logger.info(f"Answering question: {question}")
        
        # Create dependencies
        deps = QuestionAnsweringDependencies(
            context=context,
            search_results=search_results,
            code_snippets=code_snippets
        )
        
        # Run the question answering agent
        result = await self.question_answering_agent.run(question, deps, message_history)
        
        logger.info(f"Question answered: {question}")
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "new_messages": result.new_messages() if hasattr(result, 'new_messages') else []
        }

async def main():
    """Main function for the Pydantic AI agent system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pydantic AI Agent System")
    parser.add_argument("--mode", choices=["generate", "analyze", "answer"], help="Mode to run the system in")
    parser.add_argument("--prompt", help="Prompt for code generation or question")
    parser.add_argument("--language", help="Programming language")
    parser.add_argument("--framework", help="Framework to use")
    parser.add_argument("--code-file", help="File containing code to analyze")
    args = parser.parse_args()
    
    # Initialize the agent system
    system = PydanticAIAgentSystem()
    
    # Run the system based on the mode
    if args.mode == "generate" and args.prompt and args.language:
        # Generate code
        result = await system.generate_code(
            prompt=args.prompt,
            language=args.language,
            framework=args.framework
        )
        
        print("\n=== Generated Code ===\n")
        print(f"Language: {result['language']}")
        print(f"\n{result['code']}\n")
        
        if result.get("explanation"):
            print("\n=== Explanation ===\n")
            print(result["explanation"])
        
        if result.get("file_path"):
            print(f"\nSuggested file path: {result['file_path']}")
    
    elif args.mode == "analyze" and args.code_file and args.language:
        # Read code from file
        try:
            with open(args.code_file, "r") as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading code file: {e}")
            return
        
        # Analyze code
        result = await system.analyze_code(
            code=code,
            language=args.language,
            framework=args.framework
        )
        
        print("\n=== Code Analysis ===\n")
        print(result["analysis"])
        
        if result["issues"]:
            print("\n=== Issues ===\n")
            for issue in result["issues"]:
                print(f"- {issue.get('type', 'Issue')}: {issue.get('description', 'No description')}")
        
        if result["suggestions"]:
            print("\n=== Suggestions ===\n")
            for suggestion in result["suggestions"]:
                print(f"- {suggestion}")
    
    elif args.mode == "answer" and args.prompt:
        # Answer question
        result = await system.answer_question(
            question=args.prompt
        )
        
        print("\n=== Answer ===\n")
        print(result["answer"])
        
        if result.get("sources"):
            print("\n=== Sources ===\n")
            for source in result["sources"]:
                print(f"- {source}")
    
    else:
        # Interactive mode
        print("=== Pydantic AI Agent System ===")
        print("Enter 'exit' to quit")
        
        # Store conversation history for each mode
        generate_history = []
        analyze_history = []
        answer_history = []
        
        while True:
            mode = input("\nSelect mode (generate, analyze, answer): ").strip().lower()
            
            if mode == "exit":
                break
            
            if mode == "generate":
                prompt = input("Enter prompt for code generation: ")
                language = input("Enter programming language: ")
                framework = input("Enter framework (optional): ")
                
                if not framework:
                    framework = None
                
                result = await system.generate_code(
                    prompt=prompt,
                    language=language,
                    framework=framework,
                    message_history=generate_history
                )
                
                # Update conversation history
                generate_history = result.get("new_messages", [])
                
                print("\n=== Generated Code ===\n")
                print(f"Language: {result['language']}")
                print(f"\n{result['code']}\n")
                
                if result.get("explanation"):
                    print("\n=== Explanation ===\n")
                    print(result["explanation"])
                
                if result.get("file_path"):
                    print(f"\nSuggested file path: {result['file_path']}")
            
            elif mode == "analyze":
                code = input("Enter code to analyze (or file path): ")
                
                # Check if input is a file path
                if code.endswith((".py", ".js", ".ts", ".html", ".css", ".java")):
                    try:
                        with open(code, "r") as f:
                            code = f.read()
                        print(f"Read code from file: {code}")
                    except Exception as e:
                        print(f"Error reading file: {e}")
                        continue
                
                language = input("Enter programming language: ")
                framework = input("Enter framework (optional): ")
                
                if not framework:
                    framework = None
                
                result = await system.analyze_code(
                    code=code,
                    language=language,
                    framework=framework,
                    message_history=analyze_history
                )
                
                # Update conversation history
                analyze_history = result.get("new_messages", [])
                
                print("\n=== Code Analysis ===\n")
                print(result["analysis"])
                
                if result["issues"]:
                    print("\n=== Issues ===\n")
                    for issue in result["issues"]:
                        print(f"- {issue.get('type', 'Issue')}: {issue.get('description', 'No description')}")
                
                if result["suggestions"]:
                    print("\n=== Suggestions ===\n")
                    for suggestion in result["suggestions"]:
                        print(f"- {suggestion}")
            
            elif mode == "answer":
                question = input("Enter question: ")
                
                result = await system.answer_question(
                    question=question,
                    message_history=answer_history
                )
                
                # Update conversation history
                answer_history = result.get("new_messages", [])
                
                print("\n=== Answer ===\n")
                print(result["answer"])
                
                if result.get("sources"):
                    print("\n=== Sources ===\n")
                    for source in result["sources"]:
                        print(f"- {source}")
            
            else:
                print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    asyncio.run(main())

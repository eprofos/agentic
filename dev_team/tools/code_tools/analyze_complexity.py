"""
Analyze code complexity metrics.
"""

import ast
import re
from typing import Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("analyze_complexity")

def analyze_code_complexity(code: str, language: str) -> Dict[str, Any]:
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
    
    # Count functions and classes
    if language.lower() == "python":
        try:
            tree = ast.parse(code)
            
            # Use AST for accurate function and class counting
            metrics["function_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            metrics["class_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            
            # Calculate cyclomatic complexity using AST
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.Try):
                    complexity += len(node.handlers)  # Count except blocks
            
            metrics["cyclomatic_complexity"] = complexity
        except Exception as e:
            logger.error(f"Error analyzing Python code complexity with AST: {e}")
            # Fallback to simple string matching if AST parsing fails
            metrics["function_count"] = code.count("def ")
            metrics["class_count"] = code.count("class ")
            metrics["cyclomatic_complexity"] = (
                code.count("if ") + 
                code.count("elif ") + 
                code.count("for ") + 
                code.count("while ") + 
                code.count(" and ") + 
                code.count(" or ")
            )
    elif language.lower() in ["javascript", "typescript"]:
        # For JavaScript/TypeScript, use regex patterns for more accuracy
        # Count function declarations, arrow functions, and methods
        function_patterns = [
            r"function\s+\w+\s*\(",  # function declarations
            r"(?:const|let|var)\s+\w+\s*=\s*(?:function|\(.*?\)\s*=>)",  # function expressions and arrow functions
            r"\w+\s*:\s*function\s*\(",  # object methods
            r"async\s+function",  # async functions
        ]
        metrics["function_count"] = sum(len(re.findall(pattern, code)) for pattern in function_patterns)
        
        # Count class declarations including named and anonymous classes
        class_patterns = [
            r"class\s+\w+",  # named classes
            r"class\s*\{",   # anonymous classes
        ]
        metrics["class_count"] = sum(len(re.findall(pattern, code)) for pattern in class_patterns)
        
        # Calculate cyclomatic complexity
        complexity_patterns = {
            r"\bif\s*\(": 1,          # if statements
            r"\belse\s+if\b": 1,      # else if statements
            r"\bfor\s*\(": 1,         # for loops
            r"\bwhile\s*\(": 1,       # while loops
            r"\bcase\s+": 1,          # switch cases
            r"\bcatch\s*\(": 1,       # try-catch blocks
            r"&&": 1,                 # logical AND
            r"\|\|": 1,              # logical OR
            r"\?": 1,                # ternary operators
        }
        
        metrics["cyclomatic_complexity"] = sum(
            len(re.findall(pattern, code)) * weight
            for pattern, weight in complexity_patterns.items()
        )
    
    logger.info(f"Code complexity metrics: {metrics}")
    return metrics
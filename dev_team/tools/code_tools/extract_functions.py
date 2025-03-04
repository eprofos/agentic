"""
Extract function definitions from code.
"""

import ast
import re
from typing import List, Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("extract_functions")

def extract_function_definitions(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from code.
    
    Args:
        code: Code to extract function definitions from
        language: Programming language of the code
        
    Returns:
        List of dictionaries with function information
    """
    logger.info(f"Extracting function definitions for language: {language}")
    
    functions = []
    
    if language.lower() == "python":
        try:
            # Parse the Python code
            tree = ast.parse(code)
            
            # Extract function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function name
                    name = node.name
                    
                    # Get function arguments
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    
                    # Get function docstring if available
                    docstring = ast.get_docstring(node)
                    
                    # Get function body
                    body_lines = []
                    for line in code.splitlines()[node.lineno:node.end_lineno]:
                        body_lines.append(line)
                    body = "\n".join(body_lines)
                    
                    functions.append({
                        "name": name,
                        "args": args,
                        "docstring": docstring,
                        "body": body,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    })
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
        except Exception as e:
            logger.error(f"Error extracting function definitions from Python code: {e}")
    elif language.lower() == "javascript":
        # Simple regex-based extraction for JavaScript functions
        # This is a simplified approach and might not handle all cases correctly
        
        # Match function declarations: function name(...) {...}
        function_pattern = r"function\s+(\w+)\s*\((.*?)\)\s*\{([\s\S]*?)\}"
        matches = re.findall(function_pattern, code)
        
        for name, args_str, body in matches:
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            functions.append({
                "name": name,
                "args": args,
                "body": body,
                "type": "function_declaration"
            })
        
        # Match arrow functions: const name = (...) => {...}
        arrow_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*\((.*?)\)\s*=>\s*\{([\s\S]*?)\}"
        matches = re.findall(arrow_pattern, code)
        
        for name, args_str, body in matches:
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            functions.append({
                "name": name,
                "args": args,
                "body": body,
                "type": "arrow_function"
            })
    
    logger.info(f"Extracted {len(functions)} function definitions")
    return functions
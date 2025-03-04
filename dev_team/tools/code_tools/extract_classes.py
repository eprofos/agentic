"""
Extract class definitions from code.
"""

import ast
import re
from typing import List, Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("extract_classes")

def extract_class_definitions(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Extract class definitions from code.
    
    Args:
        code: Code to extract class definitions from
        language: Programming language of the code
        
    Returns:
        List of dictionaries with class information
    """
    logger.info(f"Extracting class definitions for language: {language}")
    
    classes = []
    
    if language.lower() == "python":
        try:
            # Parse the Python code
            tree = ast.parse(code)
            
            # Extract class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Get class name
                    name = node.name
                    
                    # Get base classes
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                    
                    # Get class docstring if available
                    docstring = ast.get_docstring(node)
                    
                    # Get class methods
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    # Get class body
                    body_lines = []
                    for line in code.splitlines()[node.lineno:node.end_lineno]:
                        body_lines.append(line)
                    body = "\n".join(body_lines)
                    
                    classes.append({
                        "name": name,
                        "bases": bases,
                        "docstring": docstring,
                        "methods": methods,
                        "body": body,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    })
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
        except Exception as e:
            logger.error(f"Error extracting class definitions from Python code: {e}")
    elif language.lower() == "javascript":
        # Simple regex-based extraction for JavaScript classes
        # This is a simplified approach and might not handle all cases correctly
        
        # Match class declarations: class Name {...}
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{([\s\S]*?)\}"
        matches = re.findall(class_pattern, code)
        
        for name, base, body in matches:
            # Extract methods using a simple regex
            method_pattern = r"(\w+)\s*\((.*?)\)\s*\{([\s\S]*?)\}"
            method_matches = re.findall(method_pattern, body)
            
            methods = []
            for method_name, _, _ in method_matches:
                if method_name != "constructor":
                    methods.append(method_name)
            
            classes.append({
                "name": name,
                "bases": [base] if base else [],
                "methods": methods,
                "body": body
            })
    
    logger.info(f"Extracted {len(classes)} class definitions")
    return classes
"""
Code style checking tools.
"""

from typing import List, Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("style_check")

def check_code_style(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Check for code style issues.
    
    Args:
        code: Code to analyze
        language: Programming language of the code
        
    Returns:
        List of code style issues found
    """
    logger.info("Checking for code style issues...")
    
    issues = []
    
    # Simple pattern matching for common style issues
    if language.lower() == "python":
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
    elif language.lower() == "javascript":
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
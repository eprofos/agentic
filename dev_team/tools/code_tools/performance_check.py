"""
Performance issue checking tools.
"""

from typing import List, Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("performance_check")

def check_performance_issues(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Check for performance issues in the code.
    
    Args:
        code: Code to analyze
        language: Programming language of the code
        
    Returns:
        List of performance issues found
    """
    logger.info("Checking for performance issues...")
    
    issues = []
    
    # Simple pattern matching for common performance issues
    if language.lower() == "python":
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
    elif language.lower() == "javascript":
        if "for (" in code and "length" in code:
            issues.append({
                "type": "performance",
                "severity": "low",
                "description": "Consider caching array length in for loops",
                "line": code.find("for (")
            })
    
    return issues
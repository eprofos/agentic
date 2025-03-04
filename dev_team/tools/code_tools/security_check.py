"""
Security vulnerability checking tools.
"""

from typing import List, Dict, Any

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("security_check")

def check_security_vulnerabilities(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Check for security vulnerabilities in the code.
    
    Args:
        code: Code to analyze
        language: Programming language of the code
        
    Returns:
        List of security vulnerabilities found
    """
    logger.info("Checking for security vulnerabilities...")
    
    vulnerabilities = []
    
    # Simple pattern matching for common vulnerabilities
    if language.lower() == "python":
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
    elif language.lower() == "javascript":
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
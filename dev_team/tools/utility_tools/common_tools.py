"""
Common utility tools for the Pydantic AI agent system.
These tools can be used by multiple agents for common tasks.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("common_tools")

def get_current_time() -> Dict[str, str]:
    """
    Get the current time in various formats.
    
    Returns:
        Dictionary with current time in various formats
    """
    now = datetime.now()
    
    return {
        "iso": now.isoformat(),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }

def parse_json(json_str: str) -> Dict[str, Any]:
    """
    Parse a JSON string into a Python dictionary.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as a Python dictionary
        
    Raises:
        ValueError: If the JSON string is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise ValueError(f"Invalid JSON: {e}")

def format_code(code: str, language: str) -> str:
    """
    Format code according to language-specific conventions.
    
    Args:
        code: Code to format
        language: Programming language of the code
        
    Returns:
        Formatted code
    """
    # In a real implementation, this might use language-specific formatters
    # For now, we'll just return the code as is with a message
    logger.info(f"Formatting code for language: {language}")
    return code

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Markdown text containing code blocks
        
    Returns:
        List of dictionaries with language and code
    """
    # Regular expression to match markdown code blocks
    # Format: ```language\ncode\n```
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    
    # Find all code blocks
    matches = re.findall(pattern, text)
    
    # Convert matches to list of dictionaries
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language or "text",  # Default to "text" if language is not specified
            "code": code.strip()
        })
    
    logger.info(f"Extracted {len(code_blocks)} code blocks from text")
    return code_blocks

def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path is valid and exists.
    
    Args:
        file_path: File path to validate
        
    Returns:
        True if the file path is valid and exists, False otherwise
    """
    # Check if the file path is valid
    if not file_path:
        logger.warning("Empty file path")
        return False
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    # Check if the path is a file (not a directory)
    if not os.path.isfile(file_path):
        logger.warning(f"Path is not a file: {file_path}")
        return False
    
    logger.info(f"File path is valid: {file_path}")
    return True

def read_file_content(file_path: str) -> Optional[str]:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Content of the file, or None if the file does not exist or cannot be read
    """
    # Validate the file path
    if not validate_file_path(file_path):
        return None
    
    # Read the file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        logger.info(f"Read {len(content)} characters from file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def write_file_content(file_path: str, content: str) -> bool:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        
    Returns:
        True if the content was written successfully, False otherwise
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    # Write the content to the file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Wrote {len(content)} characters to file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}")
        return False

def detect_language_from_file(file_path: str) -> Optional[str]:
    """
    Detect the programming language from a file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected programming language, or None if the language could not be detected
    """
    # Get the file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Map file extensions to languages
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".html": "html",
        ".css": "css",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".rs": "rust",
        ".sh": "bash",
        ".json": "json",
        ".md": "markdown",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sql": "sql"
    }
    
    language = extension_map.get(ext)
    if language:
        logger.info(f"Detected language {language} for file: {file_path}")
    else:
        logger.warning(f"Could not detect language for file: {file_path}")
    
    return language

"""
Functions for appending content to files.
"""

import os
from typing import Union

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("append_file")

def append_to_file(file_path: str, content: Union[str, bytes], binary: bool = False) -> bool:
    """
    Append content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to append (string or bytes)
        binary: Whether to append in binary mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Appending content to file: {file_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Open file in appropriate mode
        mode = 'ab' if binary else 'a'
        with open(file_path, mode) as f:
            if isinstance(content, str) and binary:
                content = content.encode()
            elif isinstance(content, bytes) and not binary:
                content = content.decode()
            f.write(content)
        
        logger.info("Successfully appended content to file")
        return True
    except Exception as e:
        logger.error(f"Error appending to file: {e}")
        return False
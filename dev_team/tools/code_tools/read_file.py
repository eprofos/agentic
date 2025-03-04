"""
Functions for reading file content.
"""

from typing import Optional, Union

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("read_file")

def read_file(file_path: str, binary: bool = False) -> Optional[Union[str, bytes]]:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        binary: Whether to read in binary mode
        
    Returns:
        Optional[Union[str, bytes]]: File content if successful, None otherwise
    """
    logger.info(f"Reading file: {file_path}")
    
    try:
        mode = 'rb' if binary else 'r'
        with open(file_path, mode) as f:
            content = f.read()
        logger.info("Successfully read file content")
        return content
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None
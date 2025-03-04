"""
Logging module for the Pydantic AI agent system.
Provides logging functionality with configurable log levels.
"""

import logging
import sys
from typing import Optional

from utils.config.config import config

# Define log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the specified name and log level.
    
    Args:
        name: Name of the logger
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If None, uses the log level from config
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use provided log level or get from config
    level = LOG_LEVELS.get(
        log_level or config.logging.log_level, 
        logging.INFO
    )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger

# Create a default logger for the application
logger = setup_logger("pydantic_ai_agent")

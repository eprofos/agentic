"""
Configuration module for the Pydantic AI agent system.
Loads environment variables and provides configuration settings.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API."""
    api_key: str
    api_base: str
    model: str

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str

@dataclass
class Config:
    """Main configuration class for the agent system."""
    openrouter: OpenRouterConfig
    logging: LoggingConfig

def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Config: Configuration object with all settings.
    """
    # OpenRouter configuration
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.7-sonnet")
    
    # Logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    return Config(
        openrouter=OpenRouterConfig(
            api_key=openrouter_api_key,
            api_base=openrouter_api_base,
            model=openrouter_model
        ),
        logging=LoggingConfig(
            log_level=log_level
        )
    )

# Create a global config instance
config = load_config()

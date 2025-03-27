import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Custom Obsidian API"
    PROJECT_DESCRIPTION: str = (
        "A simple FastAPI application to run custom models on Obsidian Clipper."
    )
    VERSION: str = "0.1.0"

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Model cache configuration
    DEFAULT_USE_CACHE: bool = True

    # Prompt paths
    PROMPTS_DIR: str = "prompts"
    SYSTEM_PROMPT_PATH: str = os.path.join(PROMPTS_DIR, "system.txt")
    USER_PROMPT_PATH: str = os.path.join(PROMPTS_DIR, "user.txt")

    # Default generation parameters
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 1.0
    DEFAULT_MAX_TOKENS: int = 1024

    # Environment variables
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")

    def initialize(self):
        """Initialize application settings and create necessary directories"""
        # Create prompts directory if it doesn't exist
        prompts_dir = Path(self.PROMPTS_DIR)
        prompts_dir.mkdir(exist_ok=True)

        # Create default prompt files if they don't exist
        system_prompt_path = Path(self.SYSTEM_PROMPT_PATH)
        if not system_prompt_path.exists():
            with open(system_prompt_path, "w", encoding="utf-8") as f:
                f.write(
                    "You are a helpful AI assistant that processes web content and provides valuable insights."
                )

        user_prompt_path = Path(self.USER_PROMPT_PATH)
        if not user_prompt_path.exists():
            with open(user_prompt_path, "w", encoding="utf-8") as f:
                f.write(
                    "Process the following markdown content from a webpage and respond accordingly:\n\n{content}"
                )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

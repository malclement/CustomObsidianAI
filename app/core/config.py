import os
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

    # Environment variables
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

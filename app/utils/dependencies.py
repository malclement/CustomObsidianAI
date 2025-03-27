from typing import Generator

from app.core.config import settings
from app.services.completion_service import CompletionService
from app.services.model_service import ModelService

# Global instance of ModelService
_model_service = ModelService(
    default_use_cache=settings.DEFAULT_USE_CACHE,
    huggingface_token=settings.HUGGINGFACE_TOKEN,
)

# Global instance of CompletionService
_completion_service = CompletionService(
    model_service=_model_service,
    system_prompt_path=settings.SYSTEM_PROMPT_PATH,
    user_prompt_path=settings.USER_PROMPT_PATH,
)


def get_model_service() -> ModelService:
    """
    Dependency to get the ModelService instance
    """
    return _model_service


def get_completion_service() -> CompletionService:
    """
    Dependency to get the CompletionService instance
    """
    return _completion_service

from typing import Generator

from app.core.config import settings
from app.services.model_service import ModelService

# Global instance of ModelService
_model_service = ModelService(
    default_use_cache=settings.DEFAULT_USE_CACHE,
    huggingface_token=settings.HUGGINGFACE_TOKEN,
)


def get_model_service() -> ModelService:
    """
    Dependency to get the ModelService instance
    """
    return _model_service

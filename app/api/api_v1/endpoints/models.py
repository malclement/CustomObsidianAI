import time
import traceback
from typing import Any
from typing import Dict
from typing import Optional

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import status

from app.services.model_service import ModelService
from app.utils.dependencies import get_model_service
from app.utils.error_handlers import ModelIncompatibleException
from app.utils.error_handlers import ModelNotFoundException
from app.utils.logger import get_logger
from app.utils.logger import get_request_logger

router = APIRouter()
logger = get_logger()
request_logger = get_request_logger()


@router.get("/{model_name}")
async def get_model(
    model_name: str,
    request: Request,
    text: Optional[str] = Query(None, description="Text to process with the model"),
    use_cache: bool = Query(
        True, description="Whether to use cached model if available"
    ),
    model_service: ModelService = Depends(get_model_service),
) -> Dict[str, Any]:
    """
    Pull and run a model from Hugging Face.

    Args:
    model_name: The name of the model on Hugging Face Hub
    text: Optional text to process with the model
    use_cache: Whether to use cached model if available
    model_service: ModelService dependency

    Returns:
    Model information and optional processed output
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        logger.info(f"Loading model: {model_name} (use_cache={use_cache})")

        model_info = await model_service.get_model_with_processing(
            model_name=model_name, text=text, use_cache=use_cache
        )

        # Log success
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Model {model_name} loaded successfully in {process_time:.2f}ms")

        return model_info

    except Exception as e:
        # Log the error
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"Error loading model {model_name}: {str(e)} "
            f"(after {process_time:.2f}ms)"
        )

        # Check for specific error types
        error_message = str(e).lower()
        if "not found" in error_message or "404" in error_message:
            raise ModelNotFoundException(model_name)

        # Re-raise as HTTPException with better formatting
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_type": e.__class__.__name__,
                "model_name": model_name,
            },
        )


@router.get("/clear-cache")
async def clear_cache(
    request: Request, model_service: ModelService = Depends(get_model_service)
) -> Dict[str, str]:
    """
    Clear the model cache to free up memory
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # Clear cache and log
        prev_cache_size = len(model_service.model_cache)
        model_service.clear_cache()
        logger.info(f"Model cache cleared (previously had {prev_cache_size} models)")

        return {"message": f"Model cache cleared ({prev_cache_size} models removed)"}

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to clear cache: {str(e)}"},
        )


@router.get("")
async def list_models(
    request: Request, model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    List all cached models
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # Get list of cached models
        cached_models = list(model_service.model_cache.keys())

        return {"cached_models": cached_models, "count": len(cached_models)}

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to list models: {str(e)}"},
        )

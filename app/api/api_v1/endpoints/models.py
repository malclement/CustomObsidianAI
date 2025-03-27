from typing import Any
from typing import Dict
from typing import Optional

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query

from app.services.model_service import ModelService
from app.utils.dependencies import get_model_service

router = APIRouter()


@router.get("/{model_name}")
async def get_model(
    model_name: str,
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
    try:
        return await model_service.get_model_with_processing(
            model_name=model_name, text=text, use_cache=use_cache
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.get("/clear-cache")
async def clear_cache(
    model_service: ModelService = Depends(get_model_service),
) -> Dict[str, str]:
    """
    Clear the model cache to free up memory
    """
    model_service.clear_cache()
    return {"message": "Model cache cleared"}

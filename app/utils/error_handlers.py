import traceback
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.utils.logger import get_logger
from app.utils.logger import get_request_logger

logger = get_logger()
request_logger = get_request_logger()


# Custom exception classes
class ModelNotFoundException(Exception):
    """Exception raised when a model cannot be found"""

    def __init__(self, model_name: str, message: Optional[str] = None):
        self.model_name = model_name
        self.message = message or f"Model '{model_name}' not found or cannot be loaded"
        super().__init__(self.message)


class ModelIncompatibleException(Exception):
    """Exception raised when a model is not compatible with the requested operation"""

    def __init__(self, model_name: str, operation: str, message: Optional[str] = None):
        self.model_name = model_name
        self.operation = operation
        self.message = (
            message
            or f"Model '{model_name}' is not compatible with operation '{operation}'"
        )
        super().__init__(self.message)


class TokenLimitExceededException(Exception):
    """Exception raised when input exceeds token limits"""

    def __init__(
        self, token_count: int, max_tokens: int, message: Optional[str] = None
    ):
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.message = (
            message or f"Input exceeds token limit: {token_count} > {max_tokens}"
        )
        super().__init__(self.message)


class RateLimitExceededException(Exception):
    """Exception raised when rate limits are exceeded"""

    def __init__(self, message: Optional[str] = None):
        self.message = message or "Rate limit exceeded. Please try again later."
        super().__init__(self.message)


# Error response formatter
def format_error_response(
    status_code: int,
    error_type: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format a standard error response"""
    response = {
        "error": {"type": error_type, "message": message, "status_code": status_code}
    }

    if request_id:
        response["error"]["request_id"] = request_id

    if details:
        response["error"]["details"] = details

    return response


# FastAPI exception handlers
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from FastAPI"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Extract validation errors
    error_details = []
    for error in exc.errors():
        error_details.append(
            {
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )

    # Log the error
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error=exc,
        detailed=True,
    )

    # Format response
    content = format_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="validation_error",
        message="Request validation failed",
        request_id=request_id,
        details={"errors": error_details},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=content
    )


async def model_not_found_exception_handler(
    request: Request, exc: ModelNotFoundException
):
    """Handle model not found errors"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_404_NOT_FOUND,
        error=exc,
        detailed=False,
    )

    # Format response
    content = format_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        error_type="model_not_found",
        message=exc.message,
        request_id=request_id,
        details={"model_name": exc.model_name},
    )

    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)


async def model_incompatible_exception_handler(
    request: Request, exc: ModelIncompatibleException
):
    """Handle model incompatible errors"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_400_BAD_REQUEST,
        error=exc,
        detailed=False,
    )

    # Format response
    content = format_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_type="model_incompatible",
        message=exc.message,
        request_id=request_id,
        details={"model_name": exc.model_name, "operation": exc.operation},
    )

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)


async def token_limit_exceeded_exception_handler(
    request: Request, exc: TokenLimitExceededException
):
    """Handle token limit exceeded errors"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_400_BAD_REQUEST,
        error=exc,
        detailed=False,
    )

    # Format response
    content = format_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_type="token_limit_exceeded",
        message=exc.message,
        request_id=request_id,
        details={"token_count": exc.token_count, "max_tokens": exc.max_tokens},
    )

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)


async def rate_limit_exceeded_exception_handler(
    request: Request, exc: RateLimitExceededException
):
    """Handle rate limit exceeded errors"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error=exc,
        detailed=False,
    )

    # Format response
    content = format_error_response(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_type="rate_limit_exceeded",
        message=exc.message,
        request_id=request_id,
    )

    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=content)


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error with full traceback
    request_logger.log_error(
        request_id=request_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error=exc,
        detailed=True,
    )

    # Format response (without exposing internal details)
    content = format_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="server_error",
        message="An internal server error occurred",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content
    )

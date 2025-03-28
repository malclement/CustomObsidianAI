import os
from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.utils.error_handlers import general_exception_handler
from app.utils.error_handlers import model_incompatible_exception_handler
from app.utils.error_handlers import model_not_found_exception_handler
from app.utils.error_handlers import ModelIncompatibleException
from app.utils.error_handlers import ModelNotFoundException
from app.utils.error_handlers import rate_limit_exceeded_exception_handler
from app.utils.error_handlers import RateLimitExceededException
from app.utils.error_handlers import token_limit_exceeded_exception_handler
from app.utils.error_handlers import TokenLimitExceededException
from app.utils.error_handlers import validation_exception_handler
from app.utils.logger import get_logger
from app.utils.middleware import RateLimitMiddleware
from app.utils.middleware import RequestLoggingMiddleware

logger = get_logger()


def create_application() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application
    """
    # Initialize settings
    settings.initialize()

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create FastAPI application
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
    )

    # Set up CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middlewares
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_RPM,
        enable_rate_limiting=settings.ENABLE_RATE_LIMITING,
    )

    # Register exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ModelNotFoundException, model_not_found_exception_handler)
    app.add_exception_handler(
        ModelIncompatibleException, model_incompatible_exception_handler
    )
    app.add_exception_handler(
        TokenLimitExceededException, token_limit_exceeded_exception_handler
    )
    app.add_exception_handler(
        RateLimitExceededException, rate_limit_exceeded_exception_handler
    )
    app.add_exception_handler(Exception, general_exception_handler)

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["status"])
    async def root():
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "status": "running",
        }

    @app.get("/health", tags=["status"])
    async def health():
        """Health check endpoint"""
        return {
            "status": "ok",
            "version": settings.VERSION,
            "environment": os.environ.get("ENVIRONMENT", "development"),
        }

    logger.info(f"Starting {settings.PROJECT_NAME} version {settings.VERSION}")
    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Running server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

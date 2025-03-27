from fastapi import FastAPI

from app.api.api_v1.api import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application
    """
    # Initialize settings
    settings.initialize()

    # Create FastAPI application
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
    )

    # Include API router
    application.include_router(api_router, prefix=settings.API_V1_STR)

    @application.get("/", tags=["status"])
    async def root():
        return {"message": f"Welcome to {settings.PROJECT_NAME}"}

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )

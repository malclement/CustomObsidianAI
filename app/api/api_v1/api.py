from fastapi import APIRouter

from app.api.api_v1.endpoints import models

# Main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(models.router, prefix="/models", tags=["models"])

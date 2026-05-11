from fastapi import APIRouter
from app.schemas.response import HealthResponse, ErrorResponse
from app.dependencies.model_loader import model_loader
from app.core.config import settings

router = APIRouter()

@router.get(
    "/health", 
    response_model=HealthResponse, 
    summary="📡 Check System Connectivity",
    description=(
        "### System Monitor\n"
        "Verifies that the entire diagnostic setup is online and ready to process requests.\n\n"
        "**What this check tells you:**\n"
        "- **Status**: If everything is running correctly.\n"
        "- **Models Loaded**: Confirms whether our AI experts are ready for analysis.\n"
        "- **Hardware**: Confirms the computing engine used for processing."
    ),
    responses={
        200: {"model": HealthResponse, "description": "AI Service is Healthy"},
        500: {"model": ErrorResponse, "description": "AI Service is experiencing issues"}
    }
)
async def health():
    """Service health check with detailed model status."""
    return {
        "status": "ok" if model_loader.is_ready else "initializing",
        "model_loaded": model_loader.is_ready,
        "device": settings.DEVICE
    }


from fastapi import APIRouter
from app.dependencies.model_loader import model_loader
from typing import List

router = APIRouter()

@router.get(
    "/symptoms", 
    response_model=List[str], 
    summary="📋 Get Supported Symptoms List",
    description="Returns the full list of symptoms that the Clinical AI model is trained to recognize."
)
async def get_symptoms():
    """List all symptoms allowed for multimodal diagnosis."""
    return model_loader.symptoms_list or []

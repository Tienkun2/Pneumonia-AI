from fastapi import APIRouter
from app.dependencies.model_loader import model_loader
from app.core.config import settings
from typing import List, Dict

router = APIRouter()

@router.get(
    "/symptoms",
    response_model=List[str],
    summary="📋 Get Supported Symptoms List",
    description="Returns the full list of symptom codes that the Clinical AI model recognises."
)
async def get_symptoms():
    """List all symptom codes allowed for multimodal diagnosis."""
    return model_loader.symptoms_list or settings.SELECTED_FEATURES


@router.get(
    "/symptoms/weights",
    response_model=Dict[str, float],
    summary="📊 Get Feature Importance Weights",
    description=(
        "Returns the normalised absolute coefficient of each symptom feature "
        "from the trained Logistic Regression clinical model. "
        "Values sum to 1.0 and represent each symptom's relative contribution "
        "to the clinical probability score."
    )
)
async def get_symptom_weights():
    """Feature importance weights derived from LR coef_ (not hardcoded)."""
    return model_loader.feature_weights or {}

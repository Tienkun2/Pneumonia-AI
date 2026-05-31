from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Annotated
import time
import logging
from app.schemas.response import DiagnosisResponse, ErrorResponse
from app.services.inference_service import InferenceService, inference_service
from app.exceptions.custom_exceptions import InvalidInputException, PredictionException
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/predict", 
    response_model=DiagnosisResponse, 
    summary="🩺 Start New Pneumonia Assessment",
    description=(
        "### 🧠 Multimodal Diagnostic AI\n"
        "Analyzes both an X-ray image and clinical symptoms for higher accuracy.\n\n"
        "**Recognized Symptoms (provide at least one):**\n"
        "`chills`, `fatigue`, `cough`, `high_fever`, `breathlessness`, `phlegm`, `chest_pain`, `fast_heart_rate`, `rusty_sputum`, `malaise`.\n"
    ),
    responses={
        200: {"model": DiagnosisResponse, "description": "Assessment Complete"},
        400: {"model": ErrorResponse, "description": "Submission Issue (Invalid file or missing data)"},
        500: {"model": ErrorResponse, "description": "Processing Error (Please contact support)"}
    }
)
async def predict(
    file: Annotated[
        UploadFile, 
        File(
            description="📸 Upload the Patient's Chest X-ray here (supported formats: JPG, PNG).",
        )
    ],
    service: Annotated[InferenceService, Depends(lambda: inference_service)],
    symptoms: Annotated[
        str, 
        Form(
            description="📝 List the patient's symptoms, separated by commas (Example: 'fever, persistent cough, chest pain').",
            examples=["fever,cough,chest_pain"]
        )
    ] = "",
    curb65_score: Annotated[
        int,
        Form(
            description="🧮 Optional doctor-evaluated CURB-65 clinical severity score (0-5)",
            ge=0,
            le=5
        )
    ] = None,
    custom_vision_weight: Annotated[
        float,
        Form(
            description="📐 Optional custom weight for vision model (0.0 - 1.0)",
            ge=0.0,
            le=1.0
        )
    ] = None,
    custom_clinical_weight: Annotated[
        float,
        Form(
            description="📐 Optional custom weight for clinical model (0.0 - 1.0)",
            ge=0.0,
            le=1.0
        )
    ] = None,
):
    """
    Multimodal Pneumonia Prediction Endpoint.
    Validates input and measures inference performance.
    """
    request_start = time.time()
    
    # 1. Validate File Extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidInputException(f"File type not allowed. Use: {settings.ALLOWED_EXTENSIONS}")

    # 2. Validate File Size
    # Note: We need to read a bit to know size or check headers
    # For production, checking content-length header is faster but can be spoofed
    # Reading into memory to check size:
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise InvalidInputException(f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB")

    logger.info(f"Processing request: {file.filename} ({len(content)} bytes)")

    try:
        # Perform Inference
        result = service.predict(
            content, 
            symptoms, 
            curb65_score, 
            custom_vision_weight, 
            custom_clinical_weight
        )
        
        latency = (time.time() - request_start) * 1000
        logger.info(f"Prediction successful for {file.filename}. Latency: {latency:.2f}ms")
        
        return result

    except Exception as e:
        logger.error(f"Inference failed for {file.filename}: {str(e)}")
        raise PredictionException(f"Diagnosis failed: {str(e)}")


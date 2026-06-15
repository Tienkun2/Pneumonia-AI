from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Annotated, List
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

from fastapi import HTTPException
import json
from datetime import datetime, timezone

@router.post(
    "/diagnose",
    summary="🩺 Multimodal AI Pneumonia Diagnosis Endpoint",
    description="Analyzes Chest X-ray image and clinical symptoms using EfficientNet-B4 + Logistic Regression and Calibrated Fusion."
)
async def diagnose(
    service: Annotated[InferenceService, Depends(lambda: inference_service)],
    patient_id: Annotated[str, Form(description="Unique patient identifier")] = None,
    xray_image: Annotated[UploadFile, File(description="Raw Chest X-ray image (JPG/PNG)")] = None,
    symptoms: Annotated[List[str], Form(description="List of patient symptoms")] = None
):
    request_start = time.time()
    
    # 1. Validation: check missing required fields -> 422 Unprocessable Entity
    if patient_id is None or not patient_id.strip():
        raise HTTPException(status_code=422, detail="Missing required field: patient_id")
    if xray_image is None:
        raise HTTPException(status_code=422, detail="Missing required field: xray_image")
        
    # 2. Validation: check image format -> 400 Bad Request
    ext = xray_image.filename.split(".")[-1].lower() if "." in xray_image.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image format. Allowed: {settings.ALLOWED_EXTENSIONS}")
        
    # 3. Validation: check image size -> 400 Bad Request
    content = await xray_image.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB")
        
    # 4. Parse symptoms list robustly
    parsed_symptoms = []
    if symptoms is not None:
        for s in symptoms:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            if s_stripped.startswith('[') and s_stripped.endswith(']'):
                try:
                    parsed_symptoms.extend(json.loads(s_stripped))
                except Exception:
                    # Fallback parse
                    parsed_symptoms.extend([item.strip().strip("'\"") for item in s_stripped[1:-1].split(",") if item.strip()])
            else:
                parsed_symptoms.extend([item.strip() for item in s_stripped.split(",") if item.strip()])
                
    # 5. Validation: check symptoms are valid snake_case codes -> 400 Bad Request
    for sym in parsed_symptoms:
        if sym not in settings.SELECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"Invalid symptom code: {sym}. Must be one of {settings.SELECTED_FEATURES}")
            
    try:
        # Convert parsed symptoms to comma-separated string for inference service
        symptoms_str = ",".join(parsed_symptoms)
        
        # Run prediction
        predict_res = service.predict(content, symptoms_str)
        
        # Prepare response matching specification (§4.2)
        xray_block = {
            "p_img": round(predict_res["p_img"], 3),
        }
        if predict_res.get("heatmap"):
            xray_block["gradcam_overlay"] = predict_res["heatmap"]
        if predict_res.get("lung_focus_ratio") is not None:
            xray_block["lung_focus_ratio"] = predict_res["lung_focus_ratio"]

        response_data = {
            "patient_id": patient_id,
            "xray": xray_block,
            "symptom": {
                "p_sym": round(predict_res["p_sym"], 3),
                "active_symptoms": parsed_symptoms
            },
            "fusion": {
                "nudge_logodds": round(predict_res["nudge_logodds"], 3),
                "p_fused": round(predict_res["p_fused"], 3),
                "threshold": settings.TAU_IMG,
                "decision": predict_res["decision"],
                "decision_label": predict_res["decision_label"]
            },
            "model_versions": {
                "xray": "effnetb4-g4m",
                "seg": "xrv-pspnet",
                "symptom": "lr-v1",
                "fusion": "rule-v2"
            },
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        latency = (time.time() - request_start) * 1000
        logger.info(f"Diagnosis endpoint successful for patient {patient_id}. Latency: {latency:.2f}ms")
        return response_data
    except Exception as e:
        logger.error(f"Diagnose endpoint inference failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")



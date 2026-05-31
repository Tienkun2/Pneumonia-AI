from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Annotated
import time
import logging
from app.schemas.response import GenerateReportRequest, GenerateReportResponse, MultimodalReportResponse, ErrorResponse
from app.services.inference_service import InferenceService, inference_service
from app.services.llm_service import LLMService, llm_service
from app.exceptions.custom_exceptions import InvalidInputException, PredictionException
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/predict", 
    response_model=MultimodalReportResponse, 
    summary="🩺 Full Diagnosis & LLM Review Board Report",
    description=(
        "### 🧠 Multimodal AI Diagnosis + GenAI Expert Board Review\n"
        "1. Synchronously evaluates Chest X-ray images (Vision) and patient symptoms (Clinical).\n"
        "2. Compiles a clinical master prompt containing prediction metrics and Grad-CAM hints.\n"
        "3. Feeds the data into the fine-tuned LLM medical expert model to generate a structured consensus review report.\n"
    ),
    responses={
        200: {"model": MultimodalReportResponse, "description": "Assessment and Review Complete"},
        400: {"model": ErrorResponse, "description": "Submission Issue (Invalid file or missing data)"},
        500: {"model": ErrorResponse, "description": "Processing Error (Please contact support)"}
    }
)
async def predict_with_report(
    file: Annotated[
        UploadFile, 
        File(
            description="📸 Upload the Patient's Chest X-ray here (supported formats: JPG, PNG).",
        )
    ],
    inf_service: Annotated[InferenceService, Depends(lambda: inference_service)],
    llm_serv: Annotated[LLMService, Depends(lambda: llm_service)],
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
    Multimodal Pneumonia Prediction + LLM Diagnostic Review Report.
    Runs the vision & clinical classifiers, then passes results to the LLM adapter.
    """
    request_start = time.time()
    
    # 1. Validate File Extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidInputException(f"File type not allowed. Use: {settings.ALLOWED_EXTENSIONS}")

    # 2. Validate File Size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise InvalidInputException(f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB")

    logger.info(f"Processing report request: {file.filename} ({len(content)} bytes)")

    try:
        # Step A: Perform Vision and Clinical Prediction
        predict_res = inf_service.predict(
            content, 
            symptoms, 
            curb65_score, 
            custom_vision_weight, 
            custom_clinical_weight
        )
        
        # Step B: Pass the generated master prompt to the LLM
        master_prompt = predict_res.get("master_prompt")
        if not master_prompt:
            raise PredictionException("Failed to assemble case prompt for LLM.")

        logger.info("Sending prompt to LLM Service for medical review report...")
        report_text, is_fallback = llm_serv.generate_report(master_prompt)
        
        latency = (time.time() - request_start) * 1000
        logger.info(f"Report prediction successful for {file.filename}. Latency: {latency:.2f}ms | Fallback: {is_fallback}")
        
        # Build composite response
        return {
            **predict_res,
            "llm_report": report_text,
            "llm_fallback": is_fallback
        }

    except Exception as e:
        logger.error(f"Report inference failed for {file.filename}: {str(e)}", exc_info=True)
        raise PredictionException(f"Diagnosis and report generation failed: {str(e)}")


@router.post(
    "/generate", 
    response_model=GenerateReportResponse,
    summary="💬 Direct LLM Review Report Generation",
    description="Generates a clinical review report directly from a pre-formatted text prompt (Master Prompt).",
    responses={
        200: {"model": GenerateReportResponse, "description": "Report Generated Successfully"},
        500: {"model": ErrorResponse, "description": "LLM Execution Error"}
    }
)
async def generate_report_directly(
    request: GenerateReportRequest,
    llm_serv: Annotated[LLMService, Depends(lambda: llm_service)]
):
    """Generates LLM report directly from prompt text."""
    try:
        logger.info("Direct LLM report generation requested.")
        report_text, is_fallback = llm_serv.generate_report(request.prompt)
        return {
            "report": report_text,
            "fallback": is_fallback
        }
    except Exception as e:
        logger.error(f"Direct LLM report generation failed: {str(e)}", exc_info=True)
        raise PredictionException(f"LLM Generation failed: {str(e)}")

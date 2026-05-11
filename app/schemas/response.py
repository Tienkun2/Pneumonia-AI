from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class DiagnosisResponse(BaseModel):
    vision_probability: float = Field(
        ..., 
        description="Probability of pneumonia detected from Chest X-ray image analysis", 
        ge=0.0, 
        le=1.0,
        examples=[0.85]
    )
    clinical_probability: float = Field(
        ..., 
        description="Probability of pneumonia derived from clinical symptoms analysis", 
        ge=0.0, 
        le=1.0,
        examples=[0.72]
    )
    final_score: float = Field(
        ..., 
        description="Weighted combined Multimodal AI score (Vision + Clinical)", 
        ge=0.0, 
        le=1.0,
        examples=[0.798]
    )
    risk_level: RiskLevel = Field(
        ..., 
        description="Categorical risk assessment based on AI analysis: LOW, MEDIUM, or HIGH",
        examples=[RiskLevel.HIGH]
    )
    heatmap: Optional[str] = Field(
        None, 
        description="Base64 encoded JPEG image showing the Grad-CAM heatmap highlighting infection areas",
        examples=["/9j/4AAQSkZJRgABAQ..."]
    )
    selected_symptoms: List[str] = Field(
        default_factory=list,
        description="List of symptoms that were detected and utilized for the clinical probability analysis",
        examples=[["cough", "high_fever", "chest_pain"]]
    )
    master_prompt: Optional[str] = Field(
        None,
        description="Pre-formatted professional prompt for an LLM medical review board based on current results"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vision_probability": 0.85,
                "clinical_probability": 0.72,
                "final_score": 0.798,
                "risk_level": "HIGH",
                "heatmap": "base64_encoded_image_string_here"
            }
        }
    )

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Overall health status of the service", examples=["ok"])
    model_loaded: bool = Field(..., description="Boolean flag indicating if AI models are loaded in memory", examples=[True])
    device: str = Field(..., description="Compute device used for inference (e.g., 'cpu' or 'cuda')", examples=["cuda"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "device": "cuda"
            }
        }
    )

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Detailed error message explaining what went wrong", examples=["File type not allowed. Use: ['jpg', 'jpeg', 'png']"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid input provided"
            }
        }
    )


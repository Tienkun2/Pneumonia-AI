from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os
import torch

class Settings(BaseSettings):
    # API Metadata
    PROJECT_NAME: str = "Pneumonia Multimodal AI Diagnosis Service"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths (Robust detection relative to root)
    VISION_MODEL_PATH: str = "app/models/vision/g4_final.pth"
    CLINICAL_MODEL_PATH: str = "app/models/clinical/symptom_model_refined.pkl"
    SYMPTOMS_LIST_PATH: str = "app/models/clinical/symptoms_list_refined.pkl"
    
    # Hardware Configuration
    # Auto-detect GPU if available, otherwise CPU
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Input Validation
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB limit
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    
    # Image Preprocessing Constants
    IMAGE_SIZE: int = 512
    CENTER_CROP: int = 448
    NORM_MEAN: List[float] = [0.485, 0.456, 0.406]
    NORM_STD: List[float] = [0.229, 0.224, 0.225]
    
    # Scoring & Business Logic
    VISION_WEIGHT: float = 0.7
    CLINICAL_WEIGHT: float = 0.3
    
    HIGH_RISK_THRESHOLD: float = 0.7
    MEDIUM_RISK_THRESHOLD: float = 0.345
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()


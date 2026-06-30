from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os
import torch

class Settings(BaseSettings):
    # API Metadata
    PROJECT_NAME: str = "Pneumonia Multimodal AI Diagnosis Service"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths (Robust detection relative to root)
    VISION_MODEL_PATH: str = "app/models/vision/g4m.pth"
    CLINICAL_MODEL_PATH: str = "app/models/clinical/symptom_model_lr.pkl"
    SYMPTOMS_LIST_PATH: str = "app/models/clinical/symptoms_list.pkl"
    
    # LLM Configuration
    LLM_MODEL_PATH: str = "app/models/medical_ai_adapter/kaggle/working/medical_ai_lora_adapter"
    LLM_BASE_MODEL: str = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
    ENABLE_LLM: bool = True
    GEMINI_API_KEY: str = ""
    
    # Hardware Configuration
    # Auto-detect GPU if available, otherwise CPU
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Input Validation
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB limit
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
    MEDIUM_RISK_THRESHOLD: float = 0.35

    # New System Constants for Fusion
    SELECTED_FEATURES: List[str] = ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'phlegm', 'chest_pain', 'fast_heart_rate', 'rusty_sputum', 'malaise']
    TAU_IMG: float = 0.665
    W_SYM: float = 0.5
    CAP_UP: float = 0.4
    EPS: float = 1e-6

    # Lung segmentation & inference constants
    DILATE_PX: int = 15
    USE_TTA: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()


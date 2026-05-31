import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import joblib
import logging
import time
from app.core.config import settings

# --- GLOBAL FIX FOR GRAD-CAM ---
# EfficientNet doesn't require the same ReLU monkey-patching as DenseNet,
# but we ensure modules are configured for gradient flow.
# ---------------------------------------------

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton Class to load and manage AI models for production.
    Includes model warm-up and device management.
    """
    _instance = None
    _vision_model = None
    _clinical_model = None
    _symptoms_list = None
    _is_ready = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Private method to load and prepare models."""
        start_time = time.time()
        logger.info(f"Initializing Model Loader on device: {settings.DEVICE}")

        # Set torch threads for CPU inference efficiency
        if settings.DEVICE == "cpu":
            torch.set_num_threads(4)

        # Load Vision Model (G4_final EfficientNet-B4)
        try:
            v_model = models.efficientnet_b4(weights=None)
            # The G4 model replaces only the last linear layer
            v_model.classifier[1] = nn.Linear(v_model.classifier[1].in_features, 1)
            v_model.load_state_dict(torch.load(settings.VISION_MODEL_PATH, map_location=settings.DEVICE))
            
            # Ensure no in-place ReLU/SiLU for visualization consistency
            for module in v_model.modules():
                if isinstance(module, (nn.ReLU, nn.SiLU)):
                    module.inplace = False
            
            v_model.to(settings.DEVICE)
            v_model.eval()
            self._vision_model = v_model
            logger.info("Vision model (EfficientNet-B4) loaded.")
        except Exception as e:
            logger.error(f"CRITICAL: Vision model loading failed: {e}")

        # Load Clinical Model
        try:
            self._clinical_model = joblib.load(settings.CLINICAL_MODEL_PATH)
            self._symptoms_list = joblib.load(settings.SYMPTOMS_LIST_PATH)
            logger.info("Clinical model and metadata loaded.")
        except Exception as e:
            logger.error(f"CRITICAL: Clinical model loading failed: {e}")

        # WARM-UP (To avoid latency on first request)
        if self._vision_model:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, settings.CENTER_CROP, settings.CENTER_CROP).to(settings.DEVICE)
                _ = self._vision_model(dummy_input)
            logger.info("Vision model warm-up completed.")

        self._is_ready = (self._vision_model is not None and self._clinical_model is not None)
        logger.info(f"Model Loader initialization finished in {time.time() - start_time:.2f}s.")

    @property
    def vision_model(self):
        return self._vision_model

    @property
    def clinical_model(self):
        return self._clinical_model

    @property
    def symptoms_list(self):
        return self._symptoms_list

    @property
    def is_ready(self):
        return self._is_ready

model_loader = ModelLoader()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import joblib
import logging
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton Class to load and manage AI models for production.
    Includes model warm-up and device management.
    """
    _instance = None
    _vision_model = None
    _seg_model = None
    _ll_idx = None
    _rl_idx = None
    _clinical_model = None
    _symptoms_list = None
    _p_sym_empty = None
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

        if settings.DEVICE == "cpu":
            torch.set_num_threads(4)

        # Load Vision Model (EfficientNet-B4 g4m)
        try:
            v_model = models.efficientnet_b4(weights=None)
            v_model.classifier[1] = nn.Linear(v_model.classifier[1].in_features, 1)
            v_model.load_state_dict(torch.load(settings.VISION_MODEL_PATH, map_location=settings.DEVICE))
            for module in v_model.modules():
                if isinstance(module, (nn.ReLU, nn.SiLU)):
                    module.inplace = False
            v_model.to(settings.DEVICE)
            v_model.eval()
            self._vision_model = v_model
            logger.info("Vision model (EfficientNet-B4 g4m) loaded.")
        except Exception as e:
            logger.error(f"CRITICAL: Vision model loading failed: {e}")

        # Load PSPNet Lung Segmentation Model (torchxrayvision)
        try:
            import torchxrayvision as xrv
            seg = xrv.baseline_models.chestx_det.PSPNet()
            seg.eval().to(settings.DEVICE)
            self._seg_model = seg
            self._ll_idx = seg.targets.index("Left Lung")
            self._rl_idx = seg.targets.index("Right Lung")
            logger.info(f"PSPNet loaded. LL={self._ll_idx}, RL={self._rl_idx}")
        except Exception as e:
            logger.error(f"PSPNet loading failed: {e}. Lung masking will use CV2 fallback.")

        # Load Clinical Model & compute base logit reference
        try:
            import numpy as _np
            self._clinical_model = joblib.load(settings.CLINICAL_MODEL_PATH)
            self._symptoms_list = joblib.load(settings.SYMPTOMS_LIST_PATH)
            # Compute p_sym_empty once from model (spec §5.5)
            empty_vector = [[0] * len(settings.SELECTED_FEATURES)]
            self._p_sym_empty = float(self._clinical_model.predict_proba(empty_vector)[0][1])
            # Compute normalized feature weights from LR coef_ for UI display
            coef = self._clinical_model.coef_[0]
            abs_coef = _np.abs(coef)
            total = abs_coef.sum()
            self._feature_weights = {
                feat: round(float(abs_coef[i] / total), 4)
                for i, feat in enumerate(settings.SELECTED_FEATURES)
            }
            logger.info(f"Clinical model loaded. p_sym_empty={self._p_sym_empty:.6f}. Weights computed.")
        except Exception as e:
            logger.error(f"CRITICAL: Clinical model loading failed: {e}")
            self._p_sym_empty = 0.00105  # safe fallback
            self._feature_weights = {}

        # Warm-up vision model to reduce first-request latency
        if self._vision_model:
            with torch.no_grad():
                dummy = torch.randn(1, 3, settings.CENTER_CROP, settings.CENTER_CROP).to(settings.DEVICE)
                _ = self._vision_model(dummy)
            logger.info("Vision model warm-up completed.")

        self._is_ready = (self._vision_model is not None and self._clinical_model is not None)
        logger.info(f"Model Loader ready in {time.time() - start_time:.2f}s.")

    @property
    def vision_model(self):
        return self._vision_model

    @property
    def seg_model(self):
        return self._seg_model

    @property
    def ll_idx(self):
        return self._ll_idx

    @property
    def rl_idx(self):
        return self._rl_idx

    @property
    def clinical_model(self):
        return self._clinical_model

    @property
    def symptoms_list(self):
        return self._symptoms_list

    @property
    def p_sym_empty(self):
        return self._p_sym_empty

    @property
    def feature_weights(self):
        return self._feature_weights

    @property
    def is_ready(self):
        return self._is_ready

model_loader = ModelLoader()

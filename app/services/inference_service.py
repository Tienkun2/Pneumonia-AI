import torch
import torch.nn as nn
import logging
from typing import List, Tuple
from app.dependencies.model_loader import model_loader
from app.utils.image_preprocess import preprocess_image
from app.utils.clinical_preprocess import preprocess_clinical_input, parse_comma_symptoms
from app.core.config import settings
from app.schemas.response import RiskLevel
from app.exceptions.custom_exceptions import PredictionException
from app.utils.gradcam import GradCAM
from app.utils.report_generator import generate_consultant_prompt

logger = logging.getLogger(__name__)

class InferenceService:
    @property
    def vision_model(self):
        return model_loader.vision_model

    @property
    def clinical_model(self):
        return model_loader.clinical_model

    @property
    def symptoms_list(self):
        return model_loader.symptoms_list

    def predict(
        self, 
        image_bytes: bytes, 
        symptoms_str: str, 
        curb65_score: int = None,
        custom_vision_weight: float = None,
        custom_clinical_weight: float = None
    ) -> dict:
        """Main prediction orchestration logic (CLEAN VERSION)."""
        
        if not self.vision_model or not self.clinical_model:
            logger.error("MODELS NOT LOADED")
            raise PredictionException("AI models are not initialized.")
        
        # 1. Vision Inference
        logger.info("Starting Vision Inference...")
        input_tensor, img_cropped = preprocess_image(image_bytes)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.vision_model(input_tensor.to(settings.DEVICE))
                raw_prob_vision = torch.sigmoid(logits).item()

        # Calibration: G4 optimum threshold is 0.345. We map 0.345 -> 0.5 for scaling balance
        if raw_prob_vision < 0.345:
            prob_vision = (raw_prob_vision / 0.345) * 0.5
        else:
            prob_vision = 0.5 + ((raw_prob_vision - 0.345) / (1.0 - 0.345)) * 0.5

        # 2. Grad-CAM Visualization
        heatmap_b64 = None
        gradcam_err = None
        try:
            logger.info(f"Targeting logic for Grad-CAM (Raw: {raw_prob_vision:.4f}, Calibrated: {prob_vision:.4f})...")
            # For EfficientNet-B4, features[8][0] is the last conv layer (leaf layer) before pooling
            gcam = GradCAM(self.vision_model, self.vision_model.features[8][0])
            heatmap_b64, gradcam_err = gcam.generate(input_tensor, img_cropped)
            
            if heatmap_b64:
                heatmap_b64 = f"data:image/jpeg;base64,{heatmap_b64}"
                logger.info("GRAD-CAM GENERATED SUCCESSFULLY")
            else:
                logger.warning(f"GRAD-CAM RETURNED NONE. Error: {gradcam_err}")
        except Exception as e:
            import traceback
            gradcam_err = f"Outer error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"CRITICAL GRAD-CAM ERROR: {gradcam_err}")
            heatmap_b64 = None

        # 3. Clinical Inference
        selected_symptoms = parse_comma_symptoms(symptoms_str)
        input_vector = preprocess_clinical_input(selected_symptoms, self.symptoms_list)
        clinical_probs = self.clinical_model.predict_proba([input_vector])
        prob_clinical = clinical_probs[0][1]

        # 4. Determine Weights (Trọng số)
        if custom_vision_weight is not None and custom_clinical_weight is not None:
            # Custom clinician weights
            total = custom_vision_weight + custom_clinical_weight
            if total > 0:
                vision_weight = custom_vision_weight / total
                clinical_weight = custom_clinical_weight / total
                logger.info(f"Using custom clinician weights: Vision={vision_weight:.2f}, Clinical={clinical_weight:.2f}")
            else:
                vision_weight = settings.VISION_WEIGHT
                clinical_weight = settings.CLINICAL_WEIGHT
        elif curb65_score is not None and curb65_score >= 3:
            # Dynamic weights for severe clinical symptoms
            vision_weight = 0.5
            clinical_weight = 0.5
            logger.info(f"High clinical severity (CURB-65={curb65_score} >= 3). Dynamically adjusting weights to 5:5.")
        else:
            vision_weight = settings.VISION_WEIGHT
            clinical_weight = settings.CLINICAL_WEIGHT

        # 5. Extract Clinical Alerts based on symptoms
        clinical_alerts = []
        if "breathlessness" in selected_symptoms and "fast_heart_rate" in selected_symptoms:
            clinical_alerts.append("CRITICAL: Nhịp tim nhanh kèm khó thở nguy hiểm. Nguy cơ suy hô hấp cấp tính.")
        if "rusty_sputum" in selected_symptoms:
            clinical_alerts.append("WARNING: Xuất hiện đờm màu rỉ sắt. Nghi ngờ cao nhiễm khuẩn Streptococcus pneumoniae.")
        if curb65_score is not None and curb65_score >= 3:
            clinical_alerts.append(f"CRITICAL: Điểm lâm sàng CURB-65 cao ({curb65_score}/5). Kích hoạt cơ chế bảo vệ tối đa và cưỡng chế mức độ nguy cơ HIGH.")

        # 6. Result Construction
        final_score = (prob_vision * vision_weight) + (prob_clinical * clinical_weight)
        
        # Enforce risk level override for safety
        if curb65_score is not None and curb65_score >= 3:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = self._get_risk_level(final_score)

        # 7. Master Prompt Generation
        master_prompt = generate_consultant_prompt(
            prob_vision, prob_clinical, final_score, selected_symptoms, curb65_score, vision_weight, clinical_weight
        )

        return {
            "vision_probability": round(prob_vision, 4),
            "clinical_probability": round(prob_clinical, 4),
            "final_score": round(final_score, 4),
            "risk_level": risk_level,
            "heatmap": heatmap_b64,
            "selected_symptoms": selected_symptoms,
            "master_prompt": master_prompt,
            "applied_vision_weight": round(vision_weight, 2),
            "applied_clinical_weight": round(clinical_weight, 2),
            "curb65_score": curb65_score,
            "clinical_alerts": clinical_alerts,
            "gradcam_error": gradcam_err
        }

    def _get_risk_level(self, score: float) -> RiskLevel:
        if score >= settings.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif score >= settings.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

inference_service = InferenceService()

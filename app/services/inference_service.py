import torch
import torch.nn as nn
import logging
import numpy as np
import time
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
        """Main prediction orchestration (CALIBRATED FUSION VERSION)."""

        if not self.vision_model or not self.clinical_model:
            logger.error("MODELS NOT LOADED")
            raise PredictionException("AI models are not initialized.")

        # 1. Preprocess — returns (tensor, masked_pil_resized, tta_tensor, lung_mask)
        logger.info("Starting Vision Inference...")
        t0 = time.perf_counter()
        input_tensor, img_cropped, input_tensor_tta, lung_mask = preprocess_image(image_bytes)
        t1 = time.perf_counter()

        device_str = "cuda" if settings.DEVICE == "cuda" else "cpu"

        with torch.no_grad():
            with torch.amp.autocast(device_type=device_str, enabled=torch.cuda.is_available()):
                logits = self.vision_model(input_tensor.to(settings.DEVICE))
                p_img = float(torch.sigmoid(logits).item())
                p_img = float(np.nan_to_num(p_img, nan=0.5, posinf=1.0, neginf=0.0))

        # 2. TTA: average primary + horizontal-flip prediction (spec §3 USE_TTA)
        if settings.USE_TTA:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_str, enabled=torch.cuda.is_available()):
                    logits_tta = self.vision_model(input_tensor_tta.to(settings.DEVICE))
                    p_img_tta = float(torch.sigmoid(logits_tta).item())
                    p_img_tta = float(np.nan_to_num(p_img_tta, nan=0.5, posinf=1.0, neginf=0.0))
            p_img = (p_img + p_img_tta) / 2.0
            logger.info(f"TTA applied. p_img={p_img:.4f} (avg of primary + flip)")
        else:
            logger.info(f"Vision p_img={p_img:.4f}")
        t2 = time.perf_counter()

        # 3. Grad-CAM with lung mask for focus ratio
        heatmap_b64 = None
        gradcam_err = None
        lung_focus_ratio = None
        cam_metrics = {
            "location_label": "Không khu trú rõ",
            "distribution_label": "Không rõ",
            "characteristic_label": "Không khu trú rõ",
            "foci_count": 0,
            "foci": [],
            "attention_in_lung_pct": 0.0,
            "hot_area_pct": 0.0,
            "description": "",
        }
        try:
            logger.info("Generating Grad-CAM...")
            gcam = GradCAM(self.vision_model, self.vision_model.features[8])
            res_cam = gcam.generate(
                input_tensor, img_cropped, lung_mask
            )
            heatmap_b64, gradcam_err, lung_focus_ratio, metrics_dict = res_cam
            if metrics_dict:
                cam_metrics = metrics_dict
            if heatmap_b64:
                heatmap_b64 = f"data:image/jpeg;base64,{heatmap_b64}"
                logger.info(f"Grad-CAM OK. lung_focus_ratio={lung_focus_ratio:.3f}" if lung_focus_ratio is not None else "Grad-CAM OK.")
            else:
                logger.warning(f"Grad-CAM returned None. Error: {gradcam_err}")
        except Exception as e:
            import traceback
            gradcam_err = f"Outer error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"CRITICAL GRAD-CAM ERROR: {gradcam_err}")
        t3 = time.perf_counter()

        # 4. Clinical Inference
        selected_symptoms = parse_comma_symptoms(symptoms_str)
        input_vector = [1 if s in selected_symptoms else 0 for s in settings.SELECTED_FEATURES]
        p_sym = float(self.clinical_model.predict_proba([input_vector])[0][1])

        # 5. Calibrated Fusion (spec §5.5)
        eps = settings.EPS
        p_img_c = np.clip(p_img, eps, 1.0 - eps)
        p_sym_c = np.clip(p_sym, eps, 1.0 - eps)

        # base logit computed from model at startup (not hardcoded)
        p_sym_empty = model_loader.p_sym_empty
        p_sym_empty_c = np.clip(p_sym_empty, eps, 1.0 - eps)

        z_img  = np.log(p_img_c / (1.0 - p_img_c))
        z_sym  = np.log(p_sym_c / (1.0 - p_sym_c))
        base_logit = np.log(p_sym_empty_c / (1.0 - p_sym_empty_c))

        w_sym = settings.W_SYM if custom_clinical_weight is None else custom_clinical_weight
        raw_nudge = w_sym * (z_sym - base_logit)
        nudge = float(np.clip(raw_nudge, 0.0, settings.CAP_UP))   # only nudge up

        # CURB-65 nudge: bắt đầu từ score=2 (nguy cơ tử vong 9.2%), tăng 0.35 mỗi điểm, cap 1.2
        # Tách biệt với clinical nudge vì CURB-65 là severity marker độc lập với triệu chứng
        curb65_nudge = 0.0
        if curb65_score is not None and curb65_score >= 2:
            curb65_nudge = float(np.clip(0.35 * (curb65_score - 1), 0.0, 1.2))
            logger.info(f"CURB-65={curb65_score} → curb65_nudge={curb65_nudge:.3f}")

        z_fused = z_img + nudge + curb65_nudge
        p_fused = float(1.0 / (1.0 + np.exp(-z_fused)))

        decision_bool  = p_fused >= settings.TAU_IMG
        decision       = "positive" if decision_bool else "negative"
        decision_label = ("Nghi ngờ viêm phổi (Cần lưu ý lâm sàng)"
                          if decision_bool
                          else "Bình thường hoặc Nguy cơ thấp (Cần đối chiếu thêm lâm sàng)")

        # 6. Clinical Alerts
        clinical_alerts = []
        if "breathlessness" in selected_symptoms and "fast_heart_rate" in selected_symptoms:
            clinical_alerts.append("CRITICAL: Nhịp tim nhanh kèm khó thở nguy hiểm. Nguy cơ suy hô hấp cấp tính.")
        if "rusty_sputum" in selected_symptoms:
            clinical_alerts.append("WARNING: Xuất hiện đờm màu rỉ sắt. Nghi ngờ cao nhiễm khuẩn Streptococcus pneumoniae.")
        if curb65_score is not None and curb65_score >= 3:
            clinical_alerts.append(f"CRITICAL: Điểm lâm sàng CURB-65 cao ({curb65_score}/5). Kích hoạt cơ chế bảo vệ tối đa.")

        # 7. Risk level (CURB-65 override)
        if curb65_score is not None and curb65_score >= 3:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = self._get_risk_level(p_fused)

        # 8. Master Prompt for LLM
        vision_weight = 1.0 - w_sym
        master_prompt = generate_consultant_prompt(
            p_img, p_sym, p_fused, selected_symptoms, curb65_score, vision_weight, w_sym
        )

        return {
            # /predict endpoint fields
            "vision_probability":    round(p_img, 4),
            "clinical_probability":  round(p_sym, 4),
            "final_score":           round(p_fused, 4),
            "risk_level":            risk_level,
            "heatmap":               heatmap_b64,
            "selected_symptoms":     selected_symptoms,
            "master_prompt":         master_prompt,
            "applied_vision_weight": round(vision_weight, 2),
            "applied_clinical_weight": round(w_sym, 2),
            "curb65_score":          curb65_score,
            "clinical_alerts":       clinical_alerts,
            "gradcam_error":         gradcam_err,
            # New metrics fields
            "location_label":        cam_metrics.get("location_label"),
            "distribution_label":    cam_metrics.get("distribution_label"),
            "characteristic_label":  cam_metrics.get("characteristic_label"),
            "attention_in_lung_pct": cam_metrics.get("attention_in_lung_pct"),
            "hot_area_pct":          cam_metrics.get("hot_area_pct"),
            "description":           cam_metrics.get("description"),
            # /diagnose endpoint fields
            "p_img":          p_img,
            "p_sym":          p_sym,
            "p_fused":        p_fused,
            "nudge_logodds":  nudge,
            "decision":       decision,
            "decision_label": decision_label,
            "threshold":      settings.TAU_IMG,
            "lung_focus_ratio": round(lung_focus_ratio, 3) if lung_focus_ratio is not None else None,
            # Timings
            "t_pspnet": t1 - t0,
            "t_vision": t2 - t1,
            "t_cam": t3 - t2,
        }

    def _get_risk_level(self, score: float) -> RiskLevel:
        if score >= settings.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif score >= settings.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

inference_service = InferenceService()

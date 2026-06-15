import logging
from typing import List

logger = logging.getLogger(__name__)

def generate_consultant_prompt(
    vision_prob: float, 
    clinical_prob: float, 
    final_score: float, 
    symptoms: List[str],
    curb65_score: int = None,
    vision_weight: float = 0.7,
    clinical_weight: float = 0.3
) -> str:
    """
    Assembles the 'Master Prompt' based on diagnosis results for clinical review.
    This prompt is designed to be sent to a Large Language Model (LLM) for high-level validation.
    """
    
    # Simple logic to provide a generic guidance for Grad-CAM description placeholder
    gradcam_hint = "Hãy quan sát các vùng màu Đỏ/Cam trên ảnh Heatmap đính kèm."
    if vision_prob > 0.8:
        gradcam_hint += " (Vùng kích hoạt mạnh, tập trung cao độ)"
    elif vision_prob > 0.5:
        gradcam_hint += " (Vùng kích hoạt trung bình, có dấu hiệu thâm nhiễm)"
    else:
        gradcam_hint += " (Vùng kích hoạt yếu hoặc phân tán)"

    symptoms_str = ", ".join(symptoms) if symptoms else "Không có triệu chứng điển hình"
    curb_str = f"{curb65_score}/5 điểm" if curb65_score is not None else "Chưa được đánh giá trực tiếp"

    master_prompt = f"""
## PROMPT: HỘI ĐỒNG THẨM ĐỊNH AI MULTIMODAL CHẨN ĐOÁN VIÊM PHỔI

### 1. VAI TRÒ CỦA BẠN:
Bạn là một Hội đồng chuyên gia y khoa cấp cao, bao gồm 01 Bác sĩ chẩn đoán hình ảnh (Radiologist) và 01 Chuyên gia dữ liệu lâm sàng. Nhiệm vụ của bạn là thẩm định kết quả từ một hệ thống AI Multimodal (Kết hợp X-quang và Lâm sàng) cùng các thang điểm lâm sàng của Bác sĩ.

### 2. THÔNG TIN HỆ THỐNG:
- Vision AI (EfficientNet-B4, 448px, PSPNet lung-masked): P_img → sigmoid(logit).
- Clinical AI (Logistic Regression, 10 features): P_sym → predict_proba.
- Logic Tổng hợp (Calibrated Fusion – Log-odds):
    nudge = W_sym × (logit(P_sym) − logit(P_sym_empty)), cap 0..1
    P_fused = sigmoid(logit(P_img) + nudge + curb65_nudge)
  W_sym = {clinical_weight:.2f} (triệu chứng chỉ nâng xác suất, không hạ).
- Trọng số Vision AI: {vision_weight*100:.0f}%
- Trọng số Clinical AI: {clinical_weight*100:.0f}%

### 3. DỮ LIỆU CA BỆNH HIỆN TẠI:
- **Xác suất Vision AI (P_img)**: {vision_prob * 100:.1f}%
- **Xác suất Clinical AI (P_sym)**: {clinical_prob * 100:.1f}%
- **Xác suất Tổng hợp (P_fused)**: {final_score * 100:.1f}%
- **Ngưỡng quyết định (τ)**: 66.5%
- **Vùng nhận diện Grad-CAM**: {gradcam_hint}
- **Triệu chứng khai báo**: {symptoms_str}
- **Thang điểm lâm sàng CURB-65**: {curb_str}

### 4. YÊU CẦU ĐỐI VỚI HỘI ĐỒNG:
1. **Phân tích sự đồng thuận**: Đánh giá mức độ khớp nhau giữa P_img, P_sym và thang điểm CURB-65. Chỉ ra mâu thuẫn nếu có (ví dụ: Vision AI nghi ngờ cao nhưng triệu chứng ít, hoặc ngược lại).
2. **Biện giải Grad-CAM**: Dựa trên vùng kích hoạt, giải thích ý nghĩa y khoa (Silhouette sign, Hilar congestion, Air bronchogram, Infiltration...).
3. **Khuyến nghị cuối cùng & Xử trí**: Đưa ra hướng xử trí cụ thể dựa trên P_fused và CURB-65 (Điều trị ngoại trú, Nhập viện nội trú hay ICU).
4. **Đánh giá tính hợp lý của Fusion**: Với ca bệnh này, việc nudge từ triệu chứng có phù hợp không? Có nguy cơ bỏ sót (false negative) không?

**Ngôn ngữ phản hồi**: Tiếng Việt, chuyên nghiệp, khắt khe nhưng khách quan.
"""
    return master_prompt.strip()

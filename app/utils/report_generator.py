import logging
from typing import List

logger = logging.getLogger(__name__)

def generate_consultant_prompt(
    vision_prob: float, 
    clinical_prob: float, 
    final_score: float, 
    symptoms: List[str],
    curb65_score: int = None
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
- Vision AI (EfficientNet-B4, 448px): Trọng số 70%.
- Clinical AI (Random Forest): Trọng số 30%.
- Logic Tổng hợp: P_Final = (P_Vision * 0.7) + (P_Clinical * 0.3).

### 3. DỮ LIỆU CA BỆNH HIỆN TẠI:
- **Xác suất Vision AI**: {vision_prob * 100:.1f}%
- **Xác suất Clinical AI**: {clinical_prob * 100:.1f}%
- **Xác suất Tổng hợp (Final)**: {final_score * 100:.1f}%
- **Vùng nhận diện Grad-CAM**: {gradcam_hint}
- **Triệu chứng khai báo**: {symptoms_str}
- **Thang điểm lâm sàng CURB-65**: {curb_str}

### 4. YÊU CẦU ĐỐI VỚI HỘI ĐỒNG:
1. **Phân tích sự đồng thuận**: Đánh giá mức độ khớp nhau giữa Hình ảnh, Triệu chứng Lâm sàng của AI và Thang điểm CURB-65 của bác sĩ. Chỉ ra mâu thuẫn nếu có (ví dụ: AI chẩn đoán rủi ro cao nhưng điểm CURB-65 thấp).
2. **Biện giải Grad-CAM**: Dựa trên các vùng đỏ trên heatmap, giải thích ý nghĩa y khoa (ví dụ: Silhouette sign, Hilar congestion, hoặc Infiltration).
3. **Khuyến nghị cuối cùng & Xử trí**: Đưa ra hướng xử trí cụ thể dựa trên sự kết hợp giữa kết quả AI và thang điểm CURB-65 (ví dụ: Điều trị ngoại trú, Nhập viện nội trú hay ICU).
4. **Phê bình trọng số**: Với dữ liệu này, tỉ lệ 7:3 có đang thực sự an toàn cho bệnh nhân không?

**Ngôn ngữ phản hồi**: Tiếng Việt, chuyên nghiệp, khắt khe nhưng khách quan.
"""
    return master_prompt.strip()

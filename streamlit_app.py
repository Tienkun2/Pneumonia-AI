import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Pneumonia Multimodal AI", layout="wide")
st.title("🏥 Hệ Thống Chẩn Đoán Viêm Phổi Đa Phương Thức (Ultra v3)")
st.markdown("---")

# --- 2. NẠP CÁC MODEL ---
@st.cache_resource
def load_all_models():
    # A. Model X-quang (Cấu trúc EfficientNet-B4)
    v_model = models.efficientnet_b4(weights=None)
    v_model.classifier[1] = nn.Linear(v_model.classifier[1].in_features, 1)
    # Nạp bản G4 - Bản nhìn đúng phổi nhất
    v_model.load_state_dict(torch.load('app/models/vision/g4_final.pth', map_location=torch.device('cpu')))
    
    # Tắt inplace để chạy Grad-CAM không bị lỗi lưu vết gradient
    for m in v_model.modules():
        if isinstance(m, nn.SiLU):
            m.inplace = False
            
    v_model.eval()

    # B. Model Lâm sàng
    s_model = joblib.load('app/models/clinical/symptom_model_refined.pkl')
    s_list = joblib.load('app/models/clinical/symptoms_list_refined.pkl')
    
    return v_model, s_model, s_list

vision_model, clinical_model, symptoms_list = load_all_models()

# --- 3. GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📋 Triệu Chứng Lâm Sàng")
    selected = st.multiselect("Chọn các biểu hiện hô hấp hiện có:", symptoms_list)
    
    prob_clinical = 0.0
    if selected:
        input_vector = [1 if s in selected else 0 for s in symptoms_list]
        prob_clinical = clinical_model.predict_proba([input_vector])[0][1]
        st.metric("Độ tin cậy lâm sàng", f"{prob_clinical*100:.1f}%")
        st.progress(prob_clinical)

with col2:
    st.header("📸 Phân Tích X-Quang & Grad-CAM")
    uploaded_file = st.file_uploader("Tải lên ảnh ngực thẳng...", type=["jpg", "png", "jpeg"])
    
    prob_vision = 0.0
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Tiền xử lý: Sử dụng 448x448 đồng nhất với cấu hình train G4
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # --- THỰC THI DỰ ĐOÁN & GRAD-CAM ---
        target_layers = [vision_model.features[8]]
        cam = GradCAM(model=vision_model, target_layers=target_layers)
        targets = [BinaryClassifierOutputTarget(0)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Chuẩn bị ảnh hiển thị dạng mảng float 448x448 phù hợp với input của EfficientNet
        img_cropped = np.array(image.resize((448, 448))) / 255.0
        
        # Đè Heatmap lên ảnh
        cam_image = show_cam_on_image(img_cropped, grayscale_cam, use_rgb=True)
        
        with torch.no_grad():
            logits = vision_model(input_tensor)
            raw_prob = torch.sigmoid(logits).item()
            # Hiệu chỉnh (Calibration): Ngưỡng G4 tối ưu là 0.514
            if raw_prob < 0.514:
                prob_vision = (raw_prob / 0.514) * 0.5
            else:
                prob_vision = 0.5 + ((raw_prob - 0.514) / (1.0 - 0.514)) * 0.5
        
        # Hiển thị kết quả X-quang
        st.image(cam_image, caption='Giải thích AI: Vùng tập trung phân tích (Grad-CAM)', use_column_width=True)
        st.metric("Độ tin cậy X-quang (Đã hiệu chỉnh)", f"{prob_vision*100:.1f}%")
        st.progress(prob_vision)

# --- 4. TỔNG HỢP KẾT LUẬN ---
st.markdown("---")
if st.button("📊 ĐƯA RA KẾT LUẬN TỔNG HỢP", use_container_width=True):
    if not uploaded_file or not selected:
        st.warning("⚠️ Vui lòng cung cấp đủ cả Triệu chứng và Ảnh X-quang.")
    else:
        # Công thức kết hợp
        final_score = (prob_vision * 0.6) + (prob_clinical * 0.4)
        
        st.subheader("📝 Kết quả chẩn đoán cuối cùng:")
        
        if final_score >= 0.6:
            st.error(f"🔴 NGUY CƠ CAO ({final_score*100:.2f}%)")
            st.write("**Chỉ dẫn:** AI phát hiện các dấu hiệu thâm nhiễm tại vùng rốn phổi trùng khớp với triệu chứng lâm sàng. Bệnh nhân cần được bác sĩ thăm khám trực tiếp.")
        elif 0.35 <= final_score < 0.6:
            st.warning(f"🟡 CẦN THEO DÕI ({final_score*100:.2f}%)")
            st.write("**Chỉ dẫn:** Kết quả ở mức nghi ngờ. Nên thực hiện thêm xét nghiệm CRP hoặc theo dõi sát sao nhịp thở.")
        else:
            st.success(f"🟢 NGUY CƠ THẤP ({final_score*100:.2f}%)")
            st.write("**Chỉ dẫn:** Chưa thấy dấu hiệu bệnh lý phổi rõ rệt từ cả hai phương thức.")

        # --- 5. TÍCH HỢP BÁO CÁO LLM TỪ API ---
        st.markdown("---")
        st.subheader("🤖 Hội đồng Chuyên gia Y khoa AI (Qwen2.5-Medical-LoRA)")
        
        from app.utils.report_generator import generate_consultant_prompt
        import requests
        
        # Build prompt (Note: streamlit_app.py uses 0.6/0.4 weights but we pass to standard consultant prompt)
        master_prompt = generate_consultant_prompt(
            prob_vision, prob_clinical, final_score, selected
        )
        
        backend_url = "http://localhost:8000/api/v1/report/generate"
        
        with st.spinner("Đang gửi dữ liệu phân tích tới Hội đồng Chuyên gia LLM API..."):
            try:
                response = requests.post(
                    backend_url,
                    json={"prompt": master_prompt},
                    timeout=30.0
                )
                if response.status_code == 200:
                    report_data = response.json()
                    report_text = report_data.get("report", "")
                    is_fallback = report_data.get("fallback", False)
                    
                    if is_fallback:
                        st.info("ℹ️ Báo cáo được tạo ở chế độ Mô phỏng (Offline/CPU Fallback)")
                    else:
                        st.success("✅ Báo cáo được tạo trực tiếp từ mô hình Qwen2.5 LoRA (GPU)")
                    
                    st.markdown(report_text)
                else:
                    st.error(f"Không thể kết nối tới LLM API. Mã lỗi: {response.status_code}")
                    st.info("Đảm bảo API server đang chạy tại http://localhost:8000 (Chạy lệnh: `uvicorn app.main:app`)")
            except Exception as e:
                st.error(f"Lỗi kết nối tới LLM API: {str(e)}")
                st.info("Đảm bảo API server đang chạy tại http://localhost:8000 (Chạy lệnh: `uvicorn app.main:app`)")
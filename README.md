# 🏥 Pneumonia-AI (Multimodal Diagnosis System)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

*Đọc bằng ngôn ngữ khác: [🇻🇳 Tiếng Việt](#-tiếng-việt) | [🇬🇧 English](#-english)*

---

## 🇻🇳 Tiếng Việt

Một dự án AI tiên tiến hỗ trợ tự động chẩn đoán bệnh viêm phổi. Hệ thống sử dụng **cách tiếp cận đa phương thức (multimodal)**, kết hợp khả năng phân tích hình ảnh X-quang ngực (Deep Learning) và đánh giá các triệu chứng lâm sàng (Machine Learning) để đưa ra điểm số chẩn đoán cuối cùng một cách đáng tin cậy.

### 🚀 Tính Năng Nổi Bật

*   **Chẩn Đoán Đa Phương Thức**: Kết hợp kết quả từ phân tích ảnh X-quang và triệu chứng lâm sàng của bệnh nhân.
*   **Giao Diện Người Dùng (Streamlit)**: Giao diện trực quan tích hợp công nghệ bản đồ nhiệt **Grad-CAM**, giúp giải thích vùng AI tập trung phân tích, tăng độ tin cậy cho y bác sĩ.
*   **API Hiệu Suất Cao (FastAPI)**: Backend xây dựng theo chuẩn Clean Architecture, phân tách rõ ràng luồng xử lý và logic nghiệp vụ.
*   **Kiến Trúc Mô Hình AI**: 
    *   Thị giác máy tính (Vision): Sử dụng mạng **DenseNet121** được tinh chỉnh (Custom 512-neuron classifier).
    *   Lâm sàng (Clinical): Sử dụng thuật toán **Logistic Regression**.
*   **Tính Điểm Chẩn Đoán**: Kết quả dựa trên trọng số ($0.6 \times Vision + 0.4 \times Clinical$) giúp phân loại rủi ro (Cao, Trung bình, Thấp).
*   **Bảo Mật & Production-Ready**: Kiểm tra định dạng/kích thước file, Singleton model loading, và hỗ trợ Docker.

### 📂 Cấu Trúc Dự Án

```text
├── app/                     # Backend FastAPI
│   ├── main.py              # Entrypoint của API
│   ├── api/                 # API Routes (v1)
│   ├── core/                # Cấu hình môi trường (Settings)
│   ├── services/            # Logic nghiệp vụ & xử lý mô hình
│   └── utils/               # Công cụ tiền xử lý ảnh và dữ liệu
├── streamlit_app.py         # Giao diện Frontend
├── Dockerfile               # File cấu hình Docker
├── requirements.txt         # Thư viện phụ thuộc
└── .env.example             # Biến môi trường mẫu
```

### 🛠️ Cài Đặt & Chạy Dự Án

**1. Clone dự án và cài đặt môi trường:**
```bash
git clone https://github.com/Tienkun2/Pneumonia-AI.git
cd Pneumonia-AI
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**2. Chạy Frontend (Giao diện cho bác sĩ/người dùng):**
```bash
streamlit run streamlit_app.py
```

**3. Chạy Backend API (Dành cho nhà phát triển):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Chạy bằng Docker:**
```bash
docker build -t pneumonia-ai:latest .
docker run -p 8000:8000 --env-file .env pneumonia-ai:latest
```

---

## 🇬🇧 English

A high-performance, production-ready AI microservice for the automated diagnosis of pneumonia. This system utilizes a **multimodal approach**, combining deep learning vision analysis of chest X-rays with machine learning clinical symptom evaluation to provide a robust diagnostic score.

### 🚀 Key Features

*   **Multimodal Inference**: Synchronously processes both chest X-ray images (Vision) and clinical symptoms (Tabular).
*   **Intuitive UI (Streamlit)**: Includes a frontend interface with **Grad-CAM** visual explanations, making AI decisions interpretable for medical professionals.
*   **High-Performance API (FastAPI)**: Robust backend following SOLID principles and Clean Architecture.
*   **Model Architecture**: 
    *   Vision: Custom **DenseNet121** optimized for thoracic feature extraction.
    *   Clinical: Refined **Logistic Regression** model for symptom-based risk assessment.
*   **Decision Logic**: Weighted synthesis of modalities ($0.6 \times Vision + 0.4 \times Clinical$) mapped to risk categories (High, Medium, Low).
*   **Production Optimized**: Singleton model loading, GPU auto-detection, strict input validation, and Docker support.

### 📂 Project Structure

```text
├── app/                     # FastAPI Backend Directory
│   ├── main.py              # Application entrypoint
│   ├── api/                 # API Routes (v1)
│   ├── core/                # Configuration Management
│   ├── services/            # Business Logic & Model Orchestration
│   └── utils/               # Image & Clinical Preprocessing
├── streamlit_app.py         # Streamlit Frontend UI
├── Dockerfile               # Production container config
├── requirements.txt         # Python dependencies
└── .env.example             # Template for environment variables
```

### 🛠️ Installation & Setup

**1. Clone and setup environment:**
```bash
git clone https://github.com/Tienkun2/Pneumonia-AI.git
cd Pneumonia-AI
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**2. Run Frontend User Interface:**
```bash
streamlit run streamlit_app.py
```

**3. Run Backend API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Run with Docker:**
```bash
docker build -t pneumonia-ai:latest .
docker run -p 8000:8000 --env-file .env pneumonia-ai:latest
```

---

## 📄 License
This project is proprietary and intended for clinical research purposes only.

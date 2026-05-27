import requests
import io
from PIL import Image
import os
import sys

# Define base URL
BASE_URL = "http://localhost:8000/api/v1"

def create_dummy_image():
    """Helper to create a dummy 224x224 RGB image in memory."""
    img = Image.new("RGB", (224, 224), color="gray")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_generate_endpoint():
    print("\n--- Testing /report/generate Endpoint ---")
    url = f"{BASE_URL}/report/generate"
    
    # Sample Master Prompt
    sample_prompt = """
## PROMPT: HỘI ĐỒNG THẨM ĐỊNH AI MULTIMODAL CHẨN ĐOÁN VIÊM PHỔI
- Xác suất Vision AI: 82.5%
- Xác suất Clinical AI: 40.0%
- Xác suất Tổng hợp (Final): 69.8%
- Vùng nhận diện Grad-CAM: Hãy quan sát các vùng màu Đỏ/Cam trên ảnh Heatmap đính kèm. (Vùng kích hoạt mạnh, tập trung cao độ)
- Triệu chứng khai báo: cough, high_fever
    """.strip()
    
    payload = {"prompt": sample_prompt}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("  [SUCCESS] Report generated successfully!")
            print(f"Fallback Mode: {data.get('fallback')}")
            print("\nGenerated Report Preview:")
            print(data.get("report")[:400] + "...")
        else:
            print(f"  [FAILED] Response: {response.text}")
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")

def test_predict_endpoint():
    print("\n--- Testing /report/predict Endpoint ---")
    url = f"{BASE_URL}/report/predict"
    
    dummy_img = create_dummy_image()
    files = {
        "file": ("test_image.jpg", dummy_img, "image/jpeg")
    }
    data = {
        "symptoms": "cough,high_fever"
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=15)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            res_data = response.json()
            print("  [SUCCESS] Diagnosis and report complete!")
            print(f"Vision Probability: {res_data.get('vision_probability')}")
            print(f"Clinical Probability: {res_data.get('clinical_probability')}")
            print(f"Final Score: {res_data.get('final_score')}")
            print(f"Risk Level: {res_data.get('risk_level')}")
            print(f"LLM Fallback Mode: {res_data.get('llm_fallback')}")
            print("\nLLM Review Report Preview:")
            print(res_data.get("llm_report")[:400] + "...")
        else:
            print(f"  [FAILED] Response: {response.text}")
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")

if __name__ == "__main__":
    print("Starting LLM API Integration tests...")
    print(f"Target backend URL: {BASE_URL}")
    test_generate_endpoint()
    test_predict_endpoint()

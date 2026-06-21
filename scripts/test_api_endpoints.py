import requests
import io
from PIL import Image

def run_api_tests():
    base_url = "http://127.0.0.1:8000/api/v1"
    
    print("\n--- Running API Diagnose Endpoint Integration Tests ---")
    
    # Create a dummy image in memory for testing
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Test Case A: Valid request with no symptoms
    print("Test A: Valid request with empty symptoms...")
    files = {'xray_image': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {'patient_id': 'BN0412', 'symptoms': '[]'}
    
    response = requests.post(f"{base_url}/diagnose", files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(response.text.encode('ascii', 'replace').decode('ascii'))
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["patient_id"] == "BN0412"
    assert "xray" in res_json
    assert "symptom" in res_json
    assert "fusion" in res_json
    assert "model_versions" in res_json
    assert res_json["model_versions"]["seg"] == "xrv-pspnet"
    print("Test A Passed!")
    
    # Test Case B: Valid request with symptoms array
    print("\nTest B: Valid request with symptoms array...")
    files = {'xray_image': ('test.jpg', img_byte_arr, 'image/jpeg')}
    # Send symptoms as list
    data = {'patient_id': 'BN0412', 'symptoms': '["rusty_sputum", "cough"]'}
    response = requests.post(f"{base_url}/diagnose", files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(response.text.encode('ascii', 'replace').decode('ascii'))
    assert response.status_code == 200
    res_json = response.json()
    assert "rusty_sputum" in res_json["symptom"]["active_symptoms"]
    print("Test B Passed!")

    # Test Case C: Validation error 422 - Missing patient_id
    print("\nTest C: Missing patient_id (Expect 422)...")
    files = {'xray_image': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {'symptoms': '[]'}
    response = requests.post(f"{base_url}/diagnose", files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(response.text.encode('ascii', 'replace').decode('ascii'))
    assert response.status_code == 422
    print("Test C Passed!")

    # Test Case D: Validation error 400 - Invalid symptom code
    print("\nTest D: Invalid symptom code (Expect 400)...")
    files = {'xray_image': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {'patient_id': 'BN0412', 'symptoms': '["invalid_symptom_code"]'}
    response = requests.post(f"{base_url}/diagnose", files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(response.text.encode('ascii', 'replace').decode('ascii'))
    assert response.status_code == 400
    print("Test D Passed!")

    # Test Case E: Validation error 400 - Invalid image format
    print("\nTest E: Invalid image format (Expect 400)...")
    files = {'xray_image': ('test.txt', b'some text content', 'text/plain')}
    data = {'patient_id': 'BN0412', 'symptoms': '[]'}
    response = requests.post(f"{base_url}/diagnose", files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(response.text.encode('ascii', 'replace').decode('ascii'))
    assert response.status_code == 400
    print("Test E Passed!")

    print("\nSUCCESS: All API Endpoint Integration Tests passed successfully!")

if __name__ == "__main__":
    try:
        run_api_tests()
    except Exception as e:
        print(f"FAIL: {str(e)}")

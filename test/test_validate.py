import requests
import json

url = "http://localhost:8011/ktp/validate"
headers = {"X-API-Key": "ocr-ktp-royal-2026"}

mobile_data = {
    "nik": {"value": "3217061202990002", "confidence": 91.2},
    "nama": {"value": "DEDEN SUHENDAR", "confidence": 84.5},
    "tempat_lahir": {"value": "BANDUNG", "confidence": 78.0},
    "tanggal_lahir": {"value": "12-02-1999", "confidence": 80.0},
}

with open(r"test\ocr_ktp\ktp 9.jpeg", "rb") as f:
    files = {"file": ("ktp 9.jpeg", f, "image/jpeg")}
    data = {"mobile_data": json.dumps(mobile_data)}
    
    resp = requests.post(url, headers=headers, files=files, data=data)

print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

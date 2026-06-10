# API 서버가 실행 중일 때 (예: localhost:8000)
import requests

url = "http://localhost:8000/predict"
data = {
    "bed": 150,
    "room": 45,
    "doctor": 80,
    "cluster": 0,
    "med_type": "종합병원"
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())

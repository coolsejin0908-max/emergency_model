import requests

API = "http://localhost:8000"

er = {
    "er_id": "ER99",
    "name": "테스트병원",
    "current_patients": 15,
    "available_beds": 2,
    "severity_level": 3
}
resp = requests.post(f"{API}/predict", json=er)
print("예측 결과:", resp.json())

req = {
    "user_location": "서울시 종로구",
    "severity_level": 2,
    "er_list": None
}
resp2 = requests.post(f"{API}/recommend", json=req)
print("추천 결과:", resp2.json()["best_er"])

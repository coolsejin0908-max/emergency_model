from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import requests
from model import WaitingTimeModel

app = FastAPI(title="응급실 대기시간 예측 API")
model = WaitingTimeModel()

class ERInfo(BaseModel):
    er_id: str
    name: str
    current_patients: int
    available_beds: int
    severity_level: int

class PredictResponse(BaseModel):
    er_id: str
    predicted_waiting_minutes: float

class RecommendRequest(BaseModel):
    user_location: str
    severity_level: int
    er_list: Optional[List[ERInfo]] = None

def fetch_nearby_ers(location: str):
    # 실제 공공데이터 API로 대체 필요
    return [
        ERInfo(er_id="ER_01", name="서울대병원", current_patients=12, available_beds=3, severity_level=3),
        ERInfo(er_id="ER_02", name="삼성서울병원", current_patients=8, available_beds=5, severity_level=2),
        ERInfo(er_id="ER_03", name="아산병원", current_patients=20, available_beds=1, severity_level=4),
        ERInfo(er_id="ER_04", name="세브란스병원", current_patients=5, available_beds=2, severity_level=3),
    ]

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(er: ERInfo):
    waiting = model.predict(
        er.current_patients,
        er.available_beds,
        er.severity_level
    )
    return PredictResponse(er_id=er.er_id, predicted_waiting_minutes=waiting)

@app.post("/recommend")
async def recommend_endpoint(req: RecommendRequest):
    ers = req.er_list if req.er_list else fetch_nearby_ers(req.user_location)
    results = []
    for er in ers:
        wait = model.predict(
            er.current_patients,
            er.available_beds,
            req.severity_level
        )
        results.append({
            "er_id": er.er_id,
            "name": er.name,
            "predicted_waiting_minutes": wait,
            "current_patients": er.current_patients,
            "available_beds": er.available_beds,
        })
    sorted_results = sorted(results, key=lambda x: x["predicted_waiting_minutes"])
    return {"recommendations": sorted_results, "best_er": sorted_results[0] if sorted_results else None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

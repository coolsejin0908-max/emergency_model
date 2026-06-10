from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict_emergency_congestion

app = FastAPI(title="응급실 혼잡도 예측 API")

class HospitalInput(BaseModel):
    bed: float
    room: float
    doctor: float
    cluster: int
    med_type: str   # '종합병원' 또는 '병원'

@app.post("/predict")
async def predict(data: HospitalInput):
    try:
        pred, prob = predict_emergency_congestion(
            data.bed, data.room, data.doctor,
            data.cluster, data.med_type
        )
        return {
            "predicted_congestion": pred,
            "probabilities": prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "응급실 혼잡도 예측 API - /predict POST 로 요청하세요"}

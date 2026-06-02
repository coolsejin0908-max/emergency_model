import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import joblib
import numpy as np
import os

st.set_page_config(page_title="AI 응급실 대기시간 예측기", page_icon="🚑", layout="wide")

st.title("🚑 AI 기반 응급실 대기시간 예측 & 추천 서비스")
st.markdown("공공 응급의료 데이터를 기반으로 가장 빠른 응급실을 추천합니다.")

# 모델 로드 (파일이 없으면 예외 처리)
@st.cache_resource
def load_model():
    model_path = "er_waiting_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ 모델 파일({model_path})을 찾을 수 없습니다. 먼저 train_model.py를 실행해서 파일을 생성하고 GitHub에 업로드하세요.")
        return None
    return joblib.load(model_path)

model = load_model()

def predict_waiting(patients, beds, severity):
    if model is None:
        return 999
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()
    X = np.array([[patients, beds, severity, hour, dow]])
    pred = model.predict(X)[0]
    return round(float(pred), 1)

# 주변 응급실 데이터 (실제로는 공공데이터 API로 대체 가능)
def get_nearby_ers(location):
    # 예시 데이터 (실제 프로젝트에서는 여기에 API 호출)
    return [
        {"name": "서울대병원", "patients": 12, "beds": 3},
        {"name": "삼성서울병원", "patients": 8, "beds": 5},
        {"name": "아산병원", "patients": 20, "beds": 1},
        {"name": "세브란스병원", "patients": 5, "beds": 2},
    ]

with st.sidebar:
    st.header("👤 환자 정보")
    severity_map = {"경증 (1)":1, "보통 (2)":2, "중증 (3)":3, "위중 (4)":4, "심각 (5)":5}
    severity = st.selectbox("중증도", list(severity_map.keys()), index=2)
    severity_level = severity_map[severity]
    location = st.text_input("현재 위치", "서울특별시 강남구")
    st.caption("※ 예측값은 실제와 다를 수 있습니다.")

if st.button("🔍 가장 빠른 응급실 찾기", type="primary"):
    if model is None:
        st.stop()
    with st.spinner("AI가 예측하는 중..."):
        ers = get_nearby_ers(location)
        results = []
        for er in ers:
            wait = predict_waiting(er["patients"], er["beds"], severity_level)
            results.append({
                "응급실": er["name"],
                "예상 대기시간(분)": wait,
                "현재 환자 수": er["patients"],
                "가용 병상 수": er["beds"]
            })
        df = pd.DataFrame(results)
        df = df.sort_values("예상 대기시간(분)")
        best = df.iloc[0]
        st.success(f"🏥 **최적 추천:** {best['응급실']}  \n⏱️ 예상 대기시간: **{best['예상 대기시간(분)']}분**")
        st.subheader("📋 응급실별 비교")
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(df, x="응급실", y="예상 대기시간(분)", color="예상 대기시간(분)", 
                     color_continuous_scale="Reds", text="예상 대기시간(분)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🕒 기준 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

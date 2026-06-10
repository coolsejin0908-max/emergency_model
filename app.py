import matplotlib as mpl
import matplotlib.pyplot as plt

# 시스템에 설치된 나눔폰트를 matplotlib에 등록
mpl.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
# app.py
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# ---------- 페이지 설정 ----------
st.set_page_config(page_title="응급실 혼잡도 예측", layout="wide")

# ---------- 모델 및 데이터 로드 ----------
@st.cache_resource
def load_model():
    # 모델 파일 로드 (GitHub에 업로드된 경로 기준)
    rf = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler_ml.pkl")
    le_type = joblib.load("le_medical_type.pkl")
    le_scale = joblib.load("le_bed_scale.pkl")
    return rf, scaler, le_type, le_scale

@st.cache_data
def load_hospital_data():
    # 지도 표시용 CSV (미리 계산된 혼잡도 포함)
    df = pd.read_csv("emergency_hospitals.csv")
    return df

try:
    rf, scaler, le_type, le_scale = load_model()
    df_hosp = load_hospital_data()
except Exception as e:
    st.error(f"필수 파일을 불러오지 못했습니다: {e}")
    st.stop()

# ---------- 예측 함수 ----------
def predict_congestion(bed, room, doctor, cluster, med_type):
    # 파생 변수 계산
    occupancy = room / bed
    doctor_per_room = room / doctor
    is_general = 1 if med_type == "종합병원" else 0

    # 병상 규모 결정
    if bed <= 100:
        bed_scale = "소형"
    elif bed <= 300:
        bed_scale = "중형"
    else:
        bed_scale = "대형"

    # 인코딩
    med_enc = le_type.transform([med_type])[0]
    scale_enc = le_scale.transform([bed_scale])[0]

    # 특성 순서 (학습 때와 동일)
    features = [
        bed, room, doctor, cluster,
        doctor / bed,          # 의료인_병상_비율
        occupancy,
        doctor_per_room,
        is_general,
        scale_enc
    ]
    cols = [
        "병상수", "입원실수", "의료인수", "cluster",
        "의료인_병상_비율", "입원실_점유율", "의료인_당_입원실",
        "종합병원_여부", "병상_규모_encoded"
    ]
    input_df = pd.DataFrame([features], columns=cols)
    input_scaled = scaler.transform(input_df)
    pred = rf.predict(input_scaled)[0]
    prob = rf.predict_proba(input_scaled)[0]
    prob_dict = dict(zip(rf.classes_, prob))
    return pred, prob_dict

# ---------- 사이드바: 사용자 입력 ----------
st.sidebar.title("🏥 응급실 정보 입력")

# 슬라이더와 선택 위젯 (최소값/최대값은 데이터 범위 참고)
bed = st.sidebar.slider("병상 수 (개)", min_value=30, max_value=1000, value=150, step=10)
room = st.sidebar.slider("입원실 수 (개)", min_value=5, max_value=300, value=45, step=5)
doctor = st.sidebar.slider("의료인 수 (명)", min_value=5, max_value=2000, value=80, step=10)

cluster = st.sidebar.selectbox("병원 규모 유형", options=[0, 1, 2], format_func=lambda x: {0: "중소형 병원", 1: "중대형 병원", 2: "초대형 병원"}[x])

med_type = st.sidebar.radio("의료기관 종별", options=["종합병원", "병원"])

predict_button = st.sidebar.button("🚀 혼잡도 예측", type="primary")

# ---------- 메인 화면 ----------
st.title("📊 응급실 혼잡도 예측 대시보드")

# 지도 표시 (전체 너비)
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🗺️ 응급의료시설 현황 지도")
    # 기본 지도 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    # WMS 레이어 추가
    folium.WmsTileLayer(
        url="https://safemap.go.kr/openapi2/IF_0047_WMS",
        name="응급의료시설 (WMS)",
        fmt="image/png",
        layers="0",
        transparent=True,
        overlay=True,
        control=True
    ).add_to(m)

    # CSV 마커 추가
    for _, row in df_hosp.iterrows():
        # 혼잡도에 따른 색상
        color_map = {"혼잡": "red", "보통": "orange", "여유": "green"}
        color = color_map.get(row["혼잡도"], "gray")
        popup_text = f"<b>{row['사업장명']}</b><br>혼잡도: {row['혼잡도']}<br>병상수: {row['병상수']}<br>입원실수: {row['입원실수']}<br>의료인수: {row['의료인수']}"
        folium.Marker(
            location=[row["좌표정보(Y)"], row["좌표정보(X)"]],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="plus", prefix="fa"),
            tooltip=row["사업장명"]
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=900, height=600)

with col2:
    st.subheader("📋 예측 결과")
    if predict_button:
        with st.spinner("예측 중..."):
            pred, prob = predict_congestion(bed, room, doctor, cluster, med_type)
        # 결과 표시
        if pred == "혼잡":
            st.error(f"### 🔴 예측 혼잡도: **{pred}**")
        elif pred == "보통":
            st.warning(f"### 🟠 예측 혼잡도: **{pred}**")
        else:
            st.success(f"### 🟢 예측 혼잡도: **{pred}**")
        st.markdown("#### 클래스별 확률")
        for label, p in prob.items():
            st.progress(p, text=f"{label}: {p:.1%}")
    else:
        st.info("👈 왼쪽 사이드바에서 병원 정보를 입력하고 예측 버튼을 눌러주세요.")

# 추가 설명
st.markdown("---")
st.caption("※ 지도 마커 색상: 빨강(혼잡), 주황(보통), 초록(여유). WMS 레이어는 국립중앙의료원에서 제공합니다.")

# app.py - 최종 통합 버전 (히트맵 + 디버그 포함)
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import requests
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from folium.plugins import HeatMap

# ---------- 한글 폰트 설정 ----------
try:
    mpl.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rc('font', family='NanumGothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="응급실 혼잡도 예측", layout="wide")

# ---------- Secrets에서 API 키 가져오기 ----------
try:
    SERVICE_KEY = st.secrets["API_KEY"]
except:
    st.error("❌ API_KEY가 secrets.toml에 설정되지 않았습니다.")
    st.stop()

# ---------- 캐시 데이터 로드 ----------
@st.cache_resource
def load_model():
    rf = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler_ml.pkl")
    le_type = joblib.load("le_medical_type.pkl")
    le_scale = joblib.load("le_bed_scale.pkl")
    return rf, scaler, le_type, le_scale

@st.cache_data
def load_hospital_data():
    df = pd.read_csv("emergency_hospitals.csv")
    
    # ===== 디버그 정보 출력 =====
    st.write("### 🔍 디버그: CSV 로드 결과")
    st.write(f"전체 행 수: {len(df)}")
    st.write(f"컬럼 목록: {list(df.columns)}")
    if '좌표정보(X)' in df.columns and '좌표정보(Y)' in df.columns:
        st.write(f"'좌표정보(X)' NaN 개수: {df['좌표정보(X)'].isna().sum()}")
        st.write(f"'좌표정보(Y)' NaN 개수: {df['좌표정보(Y)'].isna().sum()}")
        st.write("샘플 데이터 (첫 5행):")
        st.write(df[['사업장명', '좌표정보(X)', '좌표정보(Y)']].head())
        df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
        st.write(f"NaN 제거 후 행 수: {len(df)}")
    else:
        st.error("❌ '좌표정보(X)' 또는 '좌표정보(Y)' 컬럼이 없습니다!")
    return df

# 실시간 가용 병상 정보 조회 (5분 캐시)
@st.cache_data(ttl=300)
def get_realtime_beds(stage1="서울특별시", stage2=""):
    url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
    params = {
        "ServiceKey": SERVICE_KEY,
        "STAGE1": stage1,
        "STAGE2": stage2,
        "pageNo": "1",
        "numOfRows": "500"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            result_code = root.findtext(".//resultCode")
            if result_code == "00":
                items = root.findall(".//item")
                bed_dict = {}
                for item in items:
                    hpid = item.findtext("hpid")
                    hvec = item.findtext("hvec")
                    if hpid and hvec and hvec.isdigit():
                        bed_dict[hpid] = int(hvec)
                return bed_dict
    except Exception as e:
        st.error(f"실시간 데이터 로드 실패: {e}")
    return {}

# 거리 계산 (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# 혼잡도 예측 함수
def predict_congestion(bed, room, doctor, cluster, med_type, rf, scaler, le_type, le_scale):
    occupancy = room / bed
    doctor_per_room = room / doctor
    is_general = 1 if med_type == "종합병원" else 0
    if bed <= 100:
        bed_scale = "소형"
    elif bed <= 300:
        bed_scale = "중형"
    else:
        bed_scale = "대형"
    med_enc = le_type.transform([med_type])[0]
    scale_enc = le_scale.transform([bed_scale])[0]
    features = [
        bed, room, doctor, cluster, doctor / bed,
        occupancy, doctor_per_room, is_general, scale_enc
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
    return pred, dict(zip(rf.classes_, prob))

# ---------- 메인 ----------
rf, scaler, le_type, le_scale = load_model()
df_hosp = load_hospital_data()

# 사이드바 입력
st.sidebar.title("🏥 응급실 혼잡도 예측")
mode = st.sidebar.radio("입력 방식", ["간편 모드 (질문)", "전문가 모드 (슬라이더)"])

if mode == "간편 모드 (질문)":
    size = st.sidebar.selectbox("병원 규모", ["작은 병원 (30병상 미만)", "중간 병원 (30~100병상)", "큰 병원 (100병상 이상)"])
    doc_level = st.sidebar.selectbox("의사 수", ["1~2명", "3~5명", "6명 이상"])
    if size == "작은 병원 (30병상 미만)":
        bed = 20
    elif size == "중간 병원 (30~100병상)":
        bed = 60
    else:
        bed = 200
    if doc_level == "1~2명":
        doctor = 2
    elif doc_level == "3~5명":
        doctor = 4
    else:
        doctor = 8
    room = bed * 0.3
    cluster = 0 if bed < 100 else (1 if bed < 300 else 2)
    med_type = "종합병원" if bed > 100 else "병원"
    st.sidebar.info(f"↳ 병상수 약 {bed}개, 의료인수 약 {doctor}명으로 예측합니다.")
else:
    bed = st.sidebar.slider("병상 수 (개)", 30, 1000, 150, 10)
    room = st.sidebar.slider("입원실 수 (개)", 5, 300, 45, 5)
    doctor = st.sidebar.slider("의료인 수 (명)", 5, 2000, 80, 10)
    cluster = st.sidebar.selectbox("병원 규모 유형", [0,1,2], format_func=lambda x: {0:"중소형",1:"중대형",2:"초대형"}[x])
    med_type = st.sidebar.radio("의료기관 종별", ["종합병원", "병원"])

predict_btn = st.sidebar.button("🚀 혼잡도 예측", type="primary")

# 위치 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📍 내 주변 병원 찾기")
location = streamlit_geolocation()
user_lat, user_lon = None, None
if location and 'latitude' in location:
    user_lat, user_lon = location['latitude'], location['longitude']
    st.sidebar.success("위치 정보를 가져왔습니다.")
else:
    st.sidebar.info("버튼을 눌러 현재 위치를 공유하세요.")

# 메인 화면
st.title("📊 응급실 혼잡도 예측 대시보드")

col1, col2 = st.columns([3,1])

with col1:
    st.subheader("🗺️ 응급의료시설 현황 지도 (히트맵)")
    
    # 지도 중심 설정
    if user_lat and user_lon:
        map_center = [user_lat, user_lon]
        zoom_start = 13
    else:
        map_center = [37.5665, 126.9780]
        zoom_start = 11
    
    m = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # WMS 레이어
    folium.WmsTileLayer(
        url="https://safemap.go.kr/openapi2/IF_0047_WMS",
        name="응급의료시설 (WMS)",
        fmt="image/png",
        layers="0",
        transparent=True,
        overlay=True,
        control=True
    ).add_to(m)
    
    # 주변 병원 필터링
    df_map = df_hosp.copy()
    if user_lat and user_lon:
        df_map['distance'] = df_map.apply(
            lambda row: haversine(user_lat, user_lon, row['좌표정보(Y)'], row['좌표정보(X)']), axis=1)
        df_map = df_map[df_map['distance'] <= 10]
        st.info(f"📍 반경 10km 내 {len(df_map)}개의 응급실")
    else:
        st.info(f"🗺️ 전체 {len(df_map)}개의 응급실")
    
    # ========== 히트맵 추가 ==========
    heat_data = []
    weight_map = {'혼잡': 1.0, '보통': 0.5, '여유': 0.1}
    for _, row in df_map.iterrows():
        lat = row['좌표정보(Y)']
        lon = row['좌표정보(X)']
        weight = weight_map.get(row['혼잡도'], 0)
        heat_data.append([lat, lon, weight])
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.5).add_to(m)
    # ========== 히트맵 끝 ==========
    
    # 사용자 위치 마커
    if user_lat and user_lon:
        folium.Marker(
            location=[user_lat, user_lon],
            popup="현재 위치",
            icon=folium.Icon(color="red", icon="home", prefix="fa")
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=900, height=600)

with col2:
    st.subheader("📋 예측 결과")
    if predict_btn:
        with st.spinner("예측 중..."):
            pred, prob = predict_congestion(bed, room, doctor, cluster, med_type, rf, scaler, le_type, le_scale)
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
        st.info("👈 왼쪽 사이드바에서 정보를 입력하고 예측 버튼을 눌러주세요.")

st.markdown("---")
st.caption("🔥 히트맵: 빨간색일수록 혼잡, 초록색일수록 여유. WMS 레이어는 국립중앙의료원 제공.")
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("🗺️ 응급실 위치 테스트")

# CSV 로드
df = pd.read_csv("emergency_hospitals.csv")

# 디버그 정보 출력
st.write(f"✅ 로드된 병원 수: {len(df)}")
st.write("📌 좌표 샘플 (처음 5개):")
st.write(df[['사업장명', '좌표정보(X)', '좌표정보(Y)']].head())

# 좌표 NaN 제거
df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
st.write(f"🗺️ 유효한 좌표를 가진 병원 수: {len(df)}")

# 지도 생성 (서울 중심)
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

# 마커 추가
for _, row in df.iterrows():
    folium.Marker(
        location=[row['좌표정보(Y)'], row['좌표정보(X)']],
        popup=row['사업장명'],
        icon=folium.Icon(color='red', icon='plus', prefix='fa')
    ).add_to(m)

# 지도 표시
st_folium(m, width=900, height=600)

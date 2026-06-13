import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import folium_static
import random
from datetime import datetime

# ------------------------------
# Haversine 거리 계산 함수 (km)
# ------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반경 (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ------------------------------
# 1. 페이지 설정 및 스타일
# ------------------------------
st.set_page_config(page_title="응급실 혼잡도 및 부상 기반 추천", layout="wide")
st.title("🏥 응급실 혼잡도 예측 및 부상 기반 병원 추천")
st.markdown("---")

# ------------------------------
# 2. 더미 병원 데이터 생성 (실제 데이터로 교체 가능)
# ------------------------------
@st.cache_data
def load_hospital_data():
    # 서울 일대 가상의 응급실 데이터
    hospitals = pd.DataFrame({
        "name": ["서울대학교병원", "세브란스병원", "삼성서울병원", "서울아산병원", 
                 "강남세브란스병원", "한양대학교병원", "중앙대학교병원", "경희대학교병원",
                 "가톨릭대학교 여의도성모병원", "인제대학교 상계백병원", "고려대학교 안암병원",
                 "순천향대학교 서울병원", "이화여자목동병원", "한림대학교 강동성심병원"],
        "lat": [37.5796, 37.5641, 37.5153, 37.5254, 37.4996, 
                37.5564, 37.5057, 37.5969, 37.5323, 37.6591, 
                37.5862, 37.5341, 37.5276, 37.5551],
        "lon": [126.9970, 126.9546, 127.0407, 127.0067, 127.0288,
                127.0437, 126.9588, 127.0523, 126.9356, 127.0570,
                127.0285, 127.0002, 126.8718, 127.1563],
        "specialties": [
            "외상,심장,신경,정형", "외상,심장,신경", "외상,심장,정형", "심장,신경,소아",
            "외상,정형", "외상,소아", "심장,신경", "외상,정형,신경",
            "심장,외상", "정형,소아", "신경,외상", "심장,정형",
            "소아,외상", "정형,신경"
        ],
        "total_beds": [120, 90, 110, 95, 70, 65, 80, 75, 85, 60, 90, 70, 65, 55],
        "staff_count": [150, 120, 140, 130, 90, 85, 100, 95, 110, 80, 115, 90, 85, 75]
    })
    
    # 혼잡도: 가용 병상 비율로 결정 (실시간 변화 시뮬레이션)
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 18:
        congestion_factor = 0.6   # 주간 평균 혼잡
    else:
        congestion_factor = 0.4   # 야간 여유
    
    np.random.seed(42)
    avail_beds_ratio = np.random.uniform(0.1, 0.7, len(hospitals)) + congestion_factor * 0.2
    avail_beds_ratio = np.clip(avail_beds_ratio, 0.05, 0.9)
    hospitals["available_beds"] = (hospitals["total_beds"] * avail_beds_ratio).astype(int)
    
    # 혼잡도 레벨 결정 (가용률 기준)
    def get_congestion(avail, total):
        rate = avail / total
        if rate < 0.2:
            return "혼잡", "red"
        elif rate < 0.5:
            return "보통", "orange"
        else:
            return "여유", "green"
    
    hospitals[["congestion_text", "marker_color"]] = hospitals.apply(
        lambda row: pd.Series(get_congestion(row["available_beds"], row["total_beds"])), axis=1
    )
    return hospitals

hospitals_df = load_hospital_data()

# 부상 유형별 우선 추천 전문과목 매핑
injury_specialty_map = {
    "골절": "정형",
    "심장마비/흉통": "심장",
    "뇌졸중/두부외상": "신경",
    "대형 외상(교통사고 등)": "외상",
    "소아 응급": "소아",
    "화상": "외상",
    "호흡곤란": "심장"
}
injury_options = list(injury_specialty_map.keys())

# ------------------------------
# 3. 사용자 위치 획득 (수동 입력)
# ------------------------------
st.sidebar.header("📍 사용자 위치 설정")
user_lat = st.sidebar.number_input("위도 (lat)", value=37.5665, format="%.6f", help="예: 37.5665")
user_lon = st.sidebar.number_input("경도 (lon)", value=126.9780, format="%.6f", help="예: 126.9780")
st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")

# ------------------------------
# 4. 부상 유형 선택 및 거리 필터
# ------------------------------
st.sidebar.header("🏥 추천 조건")
injury_type = st.sidebar.selectbox("부상/증상 유형", injury_options)
radius_km = st.sidebar.slider("검색 반경 (km)", min_value=1, max_value=20, value=10, step=1)

# 병원까지의 거리 계산 및 필터링 (haversine 함수 사용)
def calculate_distance(row, user_lat, user_lon):
    return haversine_distance(user_lat, user_lon, row["lat"], row["lon"])

hospitals_df["distance_km"] = hospitals_df.apply(
    lambda row: calculate_distance(row, user_lat, user_lon), axis=1
)
nearby_hospitals = hospitals_df[hospitals_df["distance_km"] <= radius_km].copy()

if nearby_hospitals.empty:
    st.warning(f"반경 {radius_km}km 내 응급실이 없습니다. 반경을 늘려보세요.")
    st.stop()

# 추천 점수 계산: 전문과목 적합성 + 혼잡도 페널티(여유일수록 가산)
required_specialty = injury_specialty_map[injury_type]

def score_hospital(row):
    specialty_list = row["specialties"].split(",")
    specialty_match = 1 if required_specialty in specialty_list else 0
    congestion_score = {"여유": 2, "보통": 1, "혼잡": 0}[row["congestion_text"]]
    distance_score = 1 / (row["distance_km"] + 0.1)
    total = specialty_match * 10 + congestion_score * 5 + distance_score
    return total

nearby_hospitals["recommend_score"] = nearby_hospitals.apply(score_hospital, axis=1)
nearby_hospitals = nearby_hospitals.sort_values("recommend_score", ascending=False)

# ------------------------------
# 5. 지도 시각화 (folium, 색상 마커)
# ------------------------------
st.subheader(f"🗺️ 반경 {radius_km}km 내 응급실 현황 (마커색 = 혼잡도)")
map_center = [user_lat, user_lon]
m = folium.Map(location=map_center, zoom_start=12)

# 사용자 위치 마커
folium.Marker(
    location=[user_lat, user_lon],
    popup="내 위치",
    icon=folium.Icon(color="blue", icon="user", prefix="fa"),
).add_to(m)

# 병원 마커 추가
for idx, row in nearby_hospitals.iterrows():
    popup_text = f"""
    <b>{row['name']}</b><br>
    거리: {row['distance_km']:.1f} km<br>
    혼잡도: {row['congestion_text']}<br>
    전체 병상: {row['total_beds']}<br>
    가용 병상: {row['available_beds']}<br>
    전문과목: {row['specialties']}<br>
    추천 점수: {row['recommend_score']:.1f}
    """
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=row["marker_color"], icon="plus", prefix="fa"),
    ).add_to(m)

folium_static(m, width=1000, height=500)

# ------------------------------
# 6. 병원 리스트 및 추천 결과 테이블
# ------------------------------
st.subheader(f"📋 '{injury_type}'에 가장 적합한 응급실 순위")
display_cols = ["name", "distance_km", "congestion_text", "total_beds", "available_beds", "specialties", "recommend_score"]
st.dataframe(
    nearby_hospitals[display_cols].style.format({"distance_km": "{:.1f}", "recommend_score": "{:.1f}"}),
    use_container_width=True
)

# ------------------------------
# 7. 전문가 모드: 혼잡도 예측 (슬라이더)
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔬 전문가 모드 (혼잡도 예측)") 이거랑 이거랑 import streamlit as st
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
import utm
from datetime import datetime

# ---------- 한글 폰트 설정 ----------
try:
    mpl.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rc('font', family='NanumGothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="응급실 혼잡도 예측 대시보드", layout="wide")

# ---------- CSS 스타일 (지진 대시보드 스타일과 비슷하게) ----------
st.markdown("""
    <style>
    .big-font { font-size: 28px !important; font-weight: bold; color: #1f77b4; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .metric-value { font-size: 32px; font-weight: bold; }
    .metric-label { font-size: 14px; color: #6c757d; }
    hr { margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)

# ---------- Secrets에서 API 키 가져오기 ----------
try:
    SERVICE_KEY = st.secrets["API_KEY"]
except:
    SERVICE_KEY = None
    st.warning("⚠️ API_KEY가 설정되지 않았습니다. 실시간 병상 정보를 사용할 수 없습니다.")

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
    if '좌표정보(X)' not in df.columns or '좌표정보(Y)' not in df.columns:
        st.error("CSV에 좌표 컬럼이 없습니다.")
        st.stop()
    # UTM → WGS84 변환 (값이 100000 이상이면 UTM)
    sample_x = df['좌표정보(X)'].iloc[0] if len(df) > 0 else 0
    if sample_x > 100000:
        st.info("🗺️ UTM 좌표를 위도/경도로 변환 중...")
        def convert_utm_row(row):
            try:
                lat, lon = utm.to_latlon(row['좌표정보(X)'], row['좌표정보(Y)'], 52, 'N')
                return lat, lon
            except:
                return None, None
        df['위도'], df['경도'] = zip(*df.apply(convert_utm_row, axis=1))
        df = df.dropna(subset=['위도', '경도'])
        df['좌표정보(Y)'] = df['위도']
        df['좌표정보(X)'] = df['경도']
        df = df.drop(columns=['위도', '경도'])
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    return df

# 실시간 가용 병상 조회
@st.cache_data(ttl=300)
def get_realtime_beds(stage1="서울특별시", stage2=""):
    if not SERVICE_KEY:
        return {}
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

# 거리 계산
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# 혼잡도 예측
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
realtime_beds = get_realtime_beds()  # 미리 로드

# ---------- 사이드바: 입력 영역 ----------
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
    st.sidebar.info(f"↳ 병상수 약 {bed}개, 의료인수 약 {doctor}명")
else:
    bed = st.sidebar.slider("병상 수 (개)", 30, 1000, 150, 10)
    room = st.sidebar.slider("입원실 수 (개)", 5, 300, 45, 5)
    doctor = st.sidebar.slider("의료인 수 (명)", 5, 2000, 80, 10)
    cluster = st.sidebar.selectbox("병원 규모 유형", [0,1,2], format_func=lambda x: {0:"중소형",1:"중대형",2:"초대형"}[x])
    med_type = st.sidebar.radio("의료기관 종별", ["종합병원", "병원"])

predict_btn = st.sidebar.button("🚀 혼잡도 예측", type="primary", use_container_width=True)

# 위치 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📍 내 위치 설정")
location = streamlit_geolocation()
if location and 'latitude' in location:
    user_lat, user_lon = location['latitude'], location['longitude']
    st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")
else:
    user_lat, user_lon = 37.5665, 126.9780
    st.sidebar.info("기본 위치(서울시청) 사용 중")

radius_km = st.sidebar.slider("분석 반경 (km)", 1, 50, 10)

# ---------- 메인 대시보드 ----------
st.title("🏥 응급실 혼잡도 예측 대시보드")
st.markdown('<div class="big-font">실시간 응급실 혼잡도 분석</div>', unsafe_allow_html=True)
st.markdown("---")

# 상단 메트릭 (KPI)
col1, col2, col3, col4 = st.columns(4)

# 전체 병원 수
total_hosp = len(df_hosp)
col1.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{total_hosp}</div>
    <div class="metric-label">전체 응급실</div>
</div>
""", unsafe_allow_html=True)

# 반경 내 병원 수
df_near = df_hosp.copy()
df_near['distance'] = df_near.apply(
    lambda row: haversine(user_lat, user_lon, row['좌표정보(Y)'], row['좌표정보(X)']), axis=1)
near_count = len(df_near[df_near['distance'] <= radius_km])
col2.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{near_count}</div>
    <div class="metric-label">반경 {radius_km}km 내 응급실</div>
</div>
""", unsafe_allow_html=True)

# 평균 혼잡도 (예: 혼잡 비율)
if '혼잡도' in df_hosp.columns:
    congestion_rate = (df_hosp['혼잡도'] == '혼잡').mean() * 100
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{congestion_rate:.1f}%</div>
        <div class="metric-label">혼잡 응급실 비율</div>
    </div>
    """, unsafe_allow_html=True)
else:
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">-</div>
        <div class="metric-label">혼잡도 정보 없음</div>
    </div>
    """, unsafe_allow_html=True)

# 실시간 가용 병상 (API 연동)
if realtime_beds:
    avg_available = sum(realtime_beds.values()) / len(realtime_beds) if realtime_beds else 0
    col4.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_available:.1f}</div>
        <div class="metric-label">평균 가용 응급실 병상</div>
    </div>
    """, unsafe_allow_html=True)
else:
    col4.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">-</div>
        <div class="metric-label">실시간 데이터 없음</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 지도와 예측 결과를 나란히 배치
col_map, col_result = st.columns([3, 1])

with col_map:
    st.subheader("🗺️ 응급의료시설 혼잡도 지도")
    map_center = [user_lat, user_lon]
    m = folium.Map(location=map_center, zoom_start=12)
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
    # 히트맵
    near_hosp = df_near[df_near['distance'] <= radius_km].copy()
    if not near_hosp.empty and '혼잡도' in near_hosp.columns:
        heat_data = []
        weight_map = {'혼잡': 1.0, '보통': 0.5, '여유': 0.1}
        for _, row in near_hosp.iterrows():
            lat = row['좌표정보(Y)']
            lon = row['좌표정보(X)']
            weight = weight_map.get(row['혼잡도'], 0)
            heat_data.append([lat, lon, weight])
        HeatMap(heat_data, radius=15, blur=10, min_opacity=0.5).add_to(m)
    # 사용자 마커
    folium.Marker(
        location=[user_lat, user_lon],
        popup="내 위치",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)
    st_folium(m, width=900, height=500)

with col_result:
    st.subheader("📋 혼잡도 예측 결과")
    if predict_btn:
        with st.spinner("예측 중..."):
            pred, prob = predict_congestion(bed, room, doctor, cluster, med_type, rf, scaler, le_type, le_scale)
        # 위험도 스타일 (지진 대시보드처럼)
        if pred == "혼잡":
            st.error("### 🔴 매우 높음")
            st.markdown(f"#### 예측 혼잡도: **{pred}**")
        elif pred == "보통":
            st.warning("### 🟠 보통")
            st.markdown(f"#### 예측 혼잡도: **{pred}**")
        else:
            st.success("### 🟢 낮음")
            st.markdown(f"#### 예측 혼잡도: **{pred}**")
        st.markdown("---")
        st.markdown("#### 혼잡도 확률 분포")
        for label, p in prob.items():
            st.progress(p, text=f"{label}: {p:.1%}")
        st.markdown(f"**예측 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("👈 왼쪽 사이드바에서 병원 정보를 입력하고 예측 버튼을 누르세요.")

st.markdown("---")
st.subheader("📋 주변 응급실 목록 (반경 내)")
if not near_hosp.empty:
    # 표시할 컬럼 선택
    show_cols = ['사업장명', '병상수', '의료인수', '혼잡도'] if '혼잡도' in near_hosp.columns else ['사업장명', '병상수', '의료인수']
    st.dataframe(near_hosp[show_cols].head(20), use_container_width=True)
else:
    st.info("주변에 응급실이 없습니다.")

st.caption("※ 히트맵: 빨강=혼잡, 초록=여유. 마커 색상은 가용 병상 수 기준. WMS 레이어는 국립중앙의료원 제공.") 합쳐줘 
expert_beds = st.sidebar.number_input("병상 수 (개)", min_value=20, max_value=300, value=150)
expert_rooms = st.sidebar.number_input("입원실 수 (개)", min_value=10, max_value=100, value=45)
expert_staff = st.sidebar.number_input("의료인 수 (명)", min_value=20, max_value=200, value=80)

# 간단한 예측 모델 (가상)
predicted_congestion_ratio = 1.0 - (expert_beds / 300 + expert_staff / 200) / 2
predicted_congestion_ratio = np.clip(predicted_congestion_ratio, 0.1, 0.9)
predicted_level = "여유" if predicted_congestion_ratio < 0.4 else ("보통" if predicted_congestion_ratio < 0.7 else "혼잡")
st.sidebar.metric("예상 가용 병상 비율", f"{predicted_congestion_ratio:.0%}")
st.sidebar.markdown(f"**예측 혼잡도:** **:{predicted_level}**")
if predicted_level == "혼잡":
    st.sidebar.error("⚠️ 매우 혼잡할 것으로 예상됩니다.")
elif predicted_level == "보통":
    st.sidebar.warning("⚠️ 보통 수준의 혼잡도가 예상됩니다.")
else:
    st.sidebar.success("✅ 여유가 예상됩니다.")

# ------------------------------
# 8. 추가 정보
# ------------------------------
st.markdown("---")
st.caption("※ 가용 병상 및 혼잡도는 현재 시간대 기반으로 시뮬레이션되었습니다. 실제 데이터는 병원별 API 연동 시 정확해집니다.")

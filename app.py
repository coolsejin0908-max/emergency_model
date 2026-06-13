import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import requests
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# ---------- 한글 폰트 설정 ----------
try:
    mpl.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rc('font', family='NanumGothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="응급실 혼잡도 예측 및 부상 기반 추천", layout="wide")

# ---------- CSS 스타일 ----------
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
    </style>
""", unsafe_allow_html=True)

# ---------- Secrets API 키 ----------
try:
    SERVICE_KEY = st.secrets["API_KEY"]
except:
    SERVICE_KEY = None

# ---------- 거리 계산 함수 ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    """두 위경도 간 거리(km) 반환, 좌표 None 처리"""
    if None in [lat1, lon1, lat2, lon2]:
        return np.inf
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ---------- 부상 유형별 전문과목 매핑 ----------
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

# ---------- 모델 및 데이터 로드 ----------
@st.cache_resource
def load_model():
    try:
        rf = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler_ml.pkl")
        le_type = joblib.load("le_medical_type.pkl")
        le_scale = joblib.load("le_bed_scale.pkl")
        return rf, scaler, le_type, le_scale
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None, None

@st.cache_data
def load_hospital_data():
    df = pd.read_csv("emergency_hospitals.csv")
    
    # 필수 컬럼 존재 확인
    required_cols = ['사업장명', '도로명주소', '병상수', '입원실수', '의료인수', 
                     'cluster', '혼잡도', '좌표정보(X)', '좌표정보(Y)']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"CSV에 필요한 컬럼 '{col}'이 없습니다.")
            st.stop()
    
    # 좌표값 숫자 변환
    df['좌표정보(X)'] = pd.to_numeric(df['좌표정보(X)'], errors='coerce')
    df['좌표정보(Y)'] = pd.to_numeric(df['좌표정보(Y)'], errors='coerce')
    
    # 유효한 좌표 범위 (대한민국 위경도)
    def is_valid_korea_coord(lat, lon):
        return (33 <= lat <= 39) and (124 <= lon <= 132)
    
    # 기존 좌표 중 유효한 것만 남김
    df['is_valid'] = df.apply(
        lambda row: is_valid_korea_coord(row['좌표정보(Y)'], row['좌표정보(X)']), 
        axis=1
    )
    valid_mask = df['is_valid']
    
    # 세션 상태에 지오코딩 결과 캐싱 (도로명주소 -> 위경도)
    if 'geocode_cache' not in st.session_state:
        st.session_state.geocode_cache = {}
    
    # 지오코딩 함수 (세션 상태 캐시 사용)
    def geocode_address(addr):
        if addr in st.session_state.geocode_cache:
            return st.session_state.geocode_cache[addr]
        try:
            geolocator = Nominatim(user_agent="hospital_map_app")
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
            location = geocode(addr, language='ko')
            if location:
                lat, lon = location.latitude, location.longitude
                st.session_state.geocode_cache[addr] = (lat, lon)
                return lat, lon
        except Exception as e:
            st.warning(f"지오코딩 실패 ({addr}): {e}")
        return None, None
    
    # 유효하지 않은 좌표는 주소로 지오코딩
    for idx in df[~valid_mask].index:
        addr = df.loc[idx, '도로명주소']
        if pd.notna(addr):
            lat, lon = geocode_address(addr)
            if lat is not None and lon is not None:
                df.loc[idx, '좌표정보(Y)'] = lat
                df.loc[idx, '좌표정보(X)'] = lon
                valid_mask[idx] = True
            else:
                # 지오코딩 실패 시 원본 좌표 그대로 (나중에 필터링)
                pass
    
    # 최종 유효 좌표만 남김
    df = df[valid_mask].copy()
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    
    # 혼잡도 컬럼이 문자열인지 확인
    if '혼잡도' in df.columns:
        df['congestion_text'] = df['혼잡도']
    else:
        df['congestion_text'] = '보통'
    
    # 전문과목 컬럼이 없으면 임시 데이터 생성 (예시)
    if 'specialties' not in df.columns:
        # 간단한 매핑: 종합병원은 모든 과목 보유, 병원은 제한적
        def infer_specialties(row):
            if row['의료기관종별명'] == '종합병원':
                return "외상,심장,신경,정형,소아"
            else:
                return "외상,정형"  # 기본값
        df['specialties'] = df.apply(infer_specialties, axis=1)
    
    return df

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
            if root.findtext(".//resultCode") == "00":
                bed_dict = {}
                for item in root.findall(".//item"):
                    hpid = item.findtext("hpid")
                    hvec = item.findtext("hvec")
                    if hpid and hvec and hvec.isdigit():
                        bed_dict[hpid] = int(hvec)
                return bed_dict
    except Exception:
        pass
    return {}

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

# ---------- 메인 실행 ----------
rf, scaler, le_type, le_scale = load_model()
df_hosp = load_hospital_data()
realtime_beds = get_realtime_beds()

st.title("🏥 응급실 혼잡도 예측 및 부상 기반 병원 추천")
st.markdown('<div class="big-font">실시간 응급실 분석 & 맞춤형 추천</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- 사이드바: 위치 및 추천 조건 ----------
st.sidebar.header("📍 내 위치 설정")
location = streamlit_geolocation()
if location and location.get('latitude') is not None and location.get('longitude') is not None:
    user_lat = location['latitude']
    user_lon = location['longitude']
    st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")
else:
    user_lat = st.sidebar.number_input("위도 (수동 입력)", value=37.5665, format="%.6f")
    user_lon = st.sidebar.number_input("경도 (수동 입력)", value=126.9780, format="%.6f")
    st.sidebar.info("기본 위치(서울시청) 사용 중")

st.sidebar.markdown("---")
st.sidebar.header("🏥 추천 조건")
injury_type = st.sidebar.selectbox("부상/증상 유형", injury_options)
radius_km = st.sidebar.slider("검색 반경 (km)", 1, 50, 10)

st.sidebar.markdown("---")
st.sidebar.header("🔬 전문가 모드 (혼잡도 예측)")
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

# ---------- KPI 메트릭 ----------
col1, col2, col3, col4 = st.columns(4)
total_hosp = len(df_hosp)
col1.markdown(f"""
<div class="metric-card"><div class="metric-value">{total_hosp}</div><div class="metric-label">전체 응급실</div></div>
""", unsafe_allow_html=True)

df_hosp['distance'] = df_hosp.apply(
    lambda row: haversine_distance(user_lat, user_lon, row['좌표정보(Y)'], row['좌표정보(X)']), 
    axis=1
)
near_count = len(df_hosp[df_hosp['distance'] <= radius_km])
col2.markdown(f"""
<div class="metric-card"><div class="metric-value">{near_count}</div><div class="metric-label">반경 {radius_km}km 내 응급실</div></div>
""", unsafe_allow_html=True)

if '혼잡도' in df_hosp.columns:
    congestion_rate = (df_hosp['혼잡도'] == '혼잡').mean() * 100
    col3.markdown(f"""
    <div class="metric-card"><div class="metric-value">{congestion_rate:.1f}%</div><div class="metric-label">혼잡 응급실 비율</div></div>
    """, unsafe_allow_html=True)
else:
    col3.markdown("<div class='metric-card'><div class='metric-value'>-</div><div class='metric-label'>혼잡도 정보 없음</div></div>", unsafe_allow_html=True)

if realtime_beds:
    avg_available = sum(realtime_beds.values()) / max(len(realtime_beds), 1)
    col4.markdown(f"""
    <div class="metric-card"><div class="metric-value">{avg_available:.1f}</div><div class="metric-label">평균 가용 응급실 병상</div></div>
    """, unsafe_allow_html=True)
else:
    col4.markdown("<div class='metric-card'><div class='metric-value'>-</div><div class='metric-label'>실시간 데이터 없음</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- 지도 및 추천 결과 ----------
col_map, col_result = st.columns([3, 1])

with col_map:
    st.subheader("🗺️ 응급실 혼잡도 및 추천 지도")
    
    # 반경 내 병원 필터링
    near_hosp = df_hosp[df_hosp['distance'] <= radius_km].copy()
    
    # 디버깅 정보 (앱 실행 시 확인)
    with st.expander("🔍 디버깅 정보 (관리자용)"):
        st.write(f"df_hosp 전체 행 수: {len(df_hosp)}")
        st.write(f"거리 계산 후 행 수: {len(near_hosp)}")
        if not near_hosp.empty:
            st.write("좌표 샘플 (첫 3개):")
            st.write(near_hosp[['사업장명', '좌표정보(Y)', '좌표정보(X)']].head(3))
        else:
            st.warning("반경 내 병원이 없습니다. 반경을 늘리거나 좌표를 확인하세요.")
    
    if near_hosp.empty:
        st.warning(f"반경 {radius_km}km 내에 응급실이 없습니다. 반경을 늘려보세요.")
    else:
        # 지도 생성
        m = folium.Map(location=[user_lat, user_lon], zoom_start=12)
        
        # WMS 레이어 추가 (오류 무시)
        try:
            folium.WmsTileLayer(
                url="https://safemap.go.kr/openapi2/IF_0047_WMS",
                name="응급의료시설 (WMS)",
                fmt="image/png", layers="0", transparent=True, overlay=True, control=True
            ).add_to(m)
        except:
            pass
        
        # 히트맵 (혼잡도 가중치)
        heat_data = []
        weight_map = {'혼잡': 1.0, '보통': 0.5, '여유': 0.1}
        for _, row in near_hosp.iterrows():
            heat_data.append([row['좌표정보(Y)'], row['좌표정보(X)'], weight_map.get(row['congestion_text'], 0)])
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, min_opacity=0.5).add_to(m)
        
        # 사용자 위치 마커
        folium.Marker(
            location=[user_lat, user_lon], 
            popup="내 위치", 
            icon=folium.Icon(color="red", icon="home", prefix="fa")
        ).add_to(m)
        
        # 추천 점수 계산
        required_specialty = injury_specialty_map[injury_type]
        
        def score_hospital(row):
            try:
                specs = str(row['specialties']).split(',')
                spec_match = 1 if required_specialty in specs else 0
                cong_val = {"여유": 2, "보통": 1, "혼잡": 0}.get(row['congestion_text'], 1)
                dist_score = 1 / (row['distance'] + 0.1)
                return spec_match * 10 + cong_val * 5 + dist_score
            except:
                return 0.0
        
        near_hosp = near_hosp.copy()
        near_hosp['recommend_score'] = near_hosp.apply(score_hospital, axis=1)
        near_hosp = near_hosp.sort_values("recommend_score", ascending=False)
        
        # 병원 마커 추가
        for _, row in near_hosp.iterrows():
            color = {"혼잡":"red", "보통":"orange", "여유":"green"}.get(row['congestion_text'], "gray")
            popup_text = f"<b>{row['사업장명']}</b><br>거리: {row['distance']:.1f} km<br>혼잡도: {row['congestion_text']}<br>추천 점수: {row['recommend_score']:.1f}"
            folium.Marker(
                location=[row['좌표정보(Y)'], row['좌표정보(X)']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon="plus", prefix="fa")
            ).add_to(m)
        
        # 지도 출력
        st_folium(m, width=900, height=500, key="hospital_map")

with col_result:
    st.subheader("📋 맞춤형 추천 결과")
    if 'near_hosp' in locals() and not near_hosp.empty:
        top = near_hosp.iloc[0]
        st.success(f"### 🏥 최우선 추천: {top['사업장명']}")
        st.write(f"**거리:** {top['distance']:.1f} km")
        st.write(f"**혼잡도:** {top['congestion_text']}")
        st.write(f"**추천 점수:** {top['recommend_score']:.1f}")
    else:
        st.warning("반경 내 응급실이 없습니다.")
    st.markdown("---")
    st.subheader("📋 혼잡도 예측 결과")
    if predict_btn and rf is not None:
        with st.spinner("예측 중..."):
            pred, prob = predict_congestion(bed, room, doctor, cluster, med_type, rf, scaler, le_type, le_scale)
        if pred == "혼잡":
            st.error("### 🔴 매우 높음")
        elif pred == "보통":
            st.warning("### 🟠 보통")
        else:
            st.success("### 🟢 낮음")
        st.markdown(f"#### 예측 혼잡도: **{pred}**")
        st.markdown("---")
        for label, p in prob.items():
            st.progress(p, text=f"{label}: {p:.1%}")
        st.markdown(f"**예측 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("👈 사이드바에서 정보를 입력하고 예측 버튼을 누르세요.")

st.markdown("---")
st.subheader("📋 주변 응급실 목록 (추천 순)")
if 'near_hosp' in locals() and not near_hosp.empty:
    show_cols = ['사업장명', 'distance', 'congestion_text', 'recommend_score']
    st.dataframe(near_hosp[show_cols].head(20).style.format({"distance": "{:.1f}", "recommend_score": "{:.1f}"}), use_container_width=True)
else:
    st.info("표시할 병원이 없습니다.")
st.caption("※ 마커 색상: 빨강(혼잡), 주황(보통), 초록(여유). 추천 점수는 부상 전문과목 적합성 + 혼잡도 + 거리 기반.")

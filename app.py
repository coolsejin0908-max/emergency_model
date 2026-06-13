# app.py
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import utm

# ---------- 한글 폰트 설정 ----------
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="응급실 혼잡도 예측 대시보드", layout="wide")

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
        # 기존 좌표 컬럼 갱신 (folium에서 사용)
        df['좌표정보(Y)'] = df['위도']
        df['좌표정보(X)'] = df['경도']
        df = df.drop(columns=['위도', '경도'])
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    # 혼잡도 컬럼이 없으면 기본값 부여
    if '혼잡도' not in df.columns:
        df['혼잡도'] = '보통'
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
    occupancy = room / bed if bed > 0 else 0
    doctor_per_room = doctor / room if room > 0 else 0
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
        bed, room, doctor, cluster, doctor / bed if bed > 0 else 0,
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
realtime_beds = get_realtime_beds()

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

# 상단 메트릭
col1, col2, col3, col4 = st.columns(4)
total_hosp = len(df_hosp)
col1.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{total_hosp}</div>
    <div class="metric-label">전체 응급실</div>
</div>
""", unsafe_allow_html=True)

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

# 평균 혼잡도
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

# 지도와 예측 결과 나란히
col_map, col_result = st.columns([3, 1])

with col_map:
    st.subheader("🗺️ 응급의료시설 혼잡도 지도")
    map_center = [user_lat, user_lon]
    m = folium.Map(location=map_center, zoom_start=12)

    # 반경 내 병원에 대해 색상 마커 추가
    near_hosp = df_near[df_near['distance'] <= radius_km].copy()
    if not near_hosp.empty:
        for _, row in near_hosp.iterrows():
            lat = row['좌표정보(Y)']
            lon = row['좌표정보(X)']
            congestion = row.get('혼잡도', '보통')
            # 색상 결정
            if congestion == '여유':
                color = 'green'
            elif congestion == '보통':
                color = 'orange'
            elif congestion == '혼잡':
                color = 'red'
            else:
                color = 'gray'
            popup_text = f"<b>{row['사업장명']}</b><br>혼잡도: {congestion}<br>병상수: {row.get('병상수', '-')}<br>의료인수: {row.get('의료인수', '-')}"
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                tooltip=f"{row['사업장명']} ({congestion})",
                icon=folium.Icon(color=color, icon='hospital', prefix='fa')
            ).add_to(m)

    # 범례 추가
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <b>🚨 응급실 혼잡도</b><br>
        <i class="fa fa-circle" style="color: red;"></i> 혼잡<br>
        <i class="fa fa-circle" style="color: orange;"></i> 보통<br>
        <i class="fa fa-circle" style="color: green;"></i> 여유
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 사용자 위치 마커
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
    show_cols = ['사업장명', '병상수', '의료인수', '혼잡도', 'distance']
    # 컬럼 존재 여부 확인
    cols_exists = [c for c in show_cols if c in near_hosp.columns]
    st.dataframe(near_hosp[cols_exists].head(20), use_container_width=True)
else:
    st.info("주변에 응급실이 없습니다.")

st.caption("※ 마커 색상: 빨강=혼잡, 주황=보통, 초록=여유. 지도는 반경 내 응급실만 표시합니다.")

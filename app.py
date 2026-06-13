import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
import folium
from streamlit_folium import folium_static
import utm
from datetime import datetime

# ------------------------------
# Haversine 거리 계산 함수 (km)
# ------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return np.inf
    R = 6371
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
# 2. 실제 병원 데이터 로드 (emergency_hospitals.csv)
# ------------------------------
@st.cache_data
def load_hospital_data():
    df = pd.read_csv("emergency_hospitals.csv")
    
    # 필수 컬럼 확인
    required_cols = ['사업장명', '도로명주소', '병상수', '입원실수', '의료인수', 
                     'cluster', '혼잡도', '좌표정보(X)', '좌표정보(Y)']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"CSV에 '{col}' 컬럼이 없습니다.")
            st.stop()
    
    # 좌표값 숫자 변환
    df['좌표정보(X)'] = pd.to_numeric(df['좌표정보(X)'], errors='coerce')
    df['좌표정보(Y)'] = pd.to_numeric(df['좌표정보(Y)'], errors='coerce')
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    
    # UTM → 위경도 변환 (좌표값이 100,000 이상이면 UTM으로 간주)
    sample_x = df['좌표정보(X)'].iloc[0] if len(df) > 0 else 0
    if sample_x > 100000:
        st.info("🗺️ UTM 좌표를 위도/경도로 변환 중...")
        def convert(row):
            try:
                lat, lon = utm.to_latlon(row['좌표정보(X)'], row['좌표정보(Y)'], 52, 'N')
                return lat, lon
            except:
                return None, None
        df[['lat', 'lon']] = df.apply(convert, axis=1, result_type='expand')
        df = df.dropna(subset=['lat', 'lon'])
        st.success(f"변환 완료! 유효한 좌표 수: {len(df)}")
    else:
        st.info("이미 위도/경도 좌표로 간주합니다.")
        df['lat'] = df['좌표정보(Y)']
        df['lon'] = df['좌표정보(X)']
        # 대한민국 범위 필터링
        df = df[(df['lat'] >= 33) & (df['lat'] <= 39) &
                (df['lon'] >= 124) & (df['lon'] <= 132)]
    
    # 컬럼명 정리 (표시용)
    df.rename(columns={'사업장명': 'name', '도로명주소': 'address', 
                       '병상수': 'total_beds', '입원실수': 'rooms', 
                       '의료인수': 'staff_count', '혼잡도': 'congestion_text'}, inplace=True)
    
    # 마커 색상 매핑
    color_map = {'혼잡': 'red', '보통': 'orange', '여유': 'green'}
    df['marker_color'] = df['congestion_text'].map(color_map).fillna('gray')
    
    # 전문과목 컬럼이 없으면 기본값 생성 (종합병원/병원 구분)
    if 'specialties' not in df.columns:
        # 의료기관종별명 컬럼이 있다고 가정 (없으면 '종합병원' 기본값)
        if '의료기관종별명' in df.columns:
            def infer_specialties(row):
                if row['의료기관종별명'] == '종합병원':
                    return "외상,심장,신경,정형,소아"
                else:
                    return "외상,정형"
        else:
            def infer_specialties(row):
                return "외상,심장,신경,정형,소아"  # 기본
        df['specialties'] = df.apply(infer_specialties, axis=1)
    
    return df

# ------------------------------
# 3. 머신러닝 모델 및 인코더 로드
# ------------------------------
@st.cache_resource
def load_ml_models():
    try:
        rf = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler_ml.pkl")
        le_type = joblib.load("le_medical_type.pkl")
        le_scale = joblib.load("le_bed_scale.pkl")
        return rf, scaler, le_type, le_scale
    except Exception as e:
        st.warning(f"모델 로드 실패: {e} (전문가 모드 예측은 비활성화됩니다.)")
        return None, None, None, None

# ------------------------------
# 4. 혼잡도 예측 함수 (실제 모델 기반)
# ------------------------------
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

# ------------------------------
# 5. 데이터 로드
# ------------------------------
hospitals_df = load_hospital_data()
rf_model, scaler_ml, le_medical, le_bed_scale = load_ml_models()

# 부상 유형 매핑
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
# 6. 사용자 위치 설정 (수동 입력)
# ------------------------------
st.sidebar.header("📍 사용자 위치 설정")
user_lat = st.sidebar.number_input("위도 (lat)", value=37.5665, format="%.6f", help="예: 37.5665 (서울시청)")
user_lon = st.sidebar.number_input("경도 (lon)", value=126.9780, format="%.6f", help="예: 126.9780")
st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")

# ------------------------------
# 7. 추천 조건 (부상 유형, 반경)
# ------------------------------
st.sidebar.header("🏥 추천 조건")
injury_type = st.sidebar.selectbox("부상/증상 유형", injury_options)
radius_km = st.sidebar.slider("검색 반경 (km)", min_value=1, max_value=50, value=10, step=1)

# 거리 계산 및 필터링
hospitals_df["distance_km"] = hospitals_df.apply(
    lambda row: haversine_distance(user_lat, user_lon, row["lat"], row["lon"]), axis=1
)
nearby_hospitals = hospitals_df[hospitals_df["distance_km"] <= radius_km].copy()

if nearby_hospitals.empty:
    st.warning(f"반경 {radius_km}km 내 응급실이 없습니다. 반경을 늘려보세요.")
    st.stop()

# 추천 점수 계산 (전문과목 적합성 + 혼잡도 가중치 + 거리)
required_specialty = injury_specialty_map[injury_type]

def score_hospital(row):
    specialty_list = row["specialties"].split(",")
    specialty_match = 1 if required_specialty in specialty_list else 0
    congestion_score = {"여유": 2, "보통": 1, "혼잡": 0}.get(row["congestion_text"], 1)
    distance_score = 1 / (row["distance_km"] + 0.1)
    total = specialty_match * 10 + congestion_score * 5 + distance_score
    return total

nearby_hospitals["recommend_score"] = nearby_hospitals.apply(score_hospital, axis=1)
nearby_hospitals = nearby_hospitals.sort_values("recommend_score", ascending=False)

# ------------------------------
# 8. 지도 시각화 (folium_static)
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
for _, row in nearby_hospitals.iterrows():
    popup_text = f"""
    <b>{row['name']}</b><br>
    거리: {row['distance_km']:.1f} km<br>
    혼잡도: {row['congestion_text']}<br>
    전체 병상: {row['total_beds']}<br>
    가용 병상: (추후 API 연동)<br>
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
# 9. 병원 리스트 출력
# ------------------------------
st.subheader(f"📋 '{injury_type}'에 가장 적합한 응급실 순위")
display_cols = ["name", "distance_km", "congestion_text", "total_beds", "specialties", "recommend_score"]
st.dataframe(
    nearby_hospitals[display_cols].style.format({"distance_km": "{:.1f}", "recommend_score": "{:.1f}"}),
    use_container_width=True
)

# ------------------------------
# 10. 전문가 모드: 실제 ML 모델로 혼잡도 예측
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔬 전문가 모드 (혼잡도 예측)")

if rf_model is not None:
    expert_beds = st.sidebar.number_input("병상 수 (개)", min_value=20, max_value=1000, value=150, step=10)
    expert_rooms = st.sidebar.number_input("입원실 수 (개)", min_value=5, max_value=300, value=45, step=5)
    expert_staff = st.sidebar.number_input("의료인 수 (명)", min_value=5, max_value=2000, value=80, step=10)
    expert_cluster = st.sidebar.selectbox("병원 규모 유형", [0,1,2], format_func=lambda x: {0:"중소형",1:"중대형",2:"초대형"}[x])
    expert_med_type = st.sidebar.radio("의료기관 종별", ["종합병원", "병원"])
    
    if st.sidebar.button("🚀 혼잡도 예측 실행", type="primary"):
        with st.spinner("예측 중..."):
            pred, prob = predict_congestion(
                expert_beds, expert_rooms, expert_staff, expert_cluster, 
                expert_med_type, rf_model, scaler_ml, le_medical, le_bed_scale
            )
        st.sidebar.markdown(f"**예측 혼잡도:** **:{pred}**")
        if pred == "혼잡":
            st.sidebar.error("⚠️ 매우 혼잡할 것으로 예상됩니다.")
        elif pred == "보통":
            st.sidebar.warning("⚠️ 보통 수준의 혼잡도가 예상됩니다.")
        else:
            st.sidebar.success("✅ 여유가 예상됩니다.")
        with st.sidebar.expander("클래스별 확률"):
            for label, p in prob.items():
                st.write(f"{label}: {p:.1%}")
else:
    st.sidebar.info("모델 파일이 없어 간단한 규칙 기반 예측을 제공합니다.")
    expert_beds = st.sidebar.number_input("병상 수 (개)", value=150)
    expert_staff = st.sidebar.number_input("의료인 수 (명)", value=80)
    predicted_ratio = 1.0 - (expert_beds / 500 + expert_staff / 300) / 2
    predicted_ratio = np.clip(predicted_ratio, 0.1, 0.9)
    predicted_level = "여유" if predicted_ratio < 0.4 else ("보통" if predicted_ratio < 0.7 else "혼잡")
    st.sidebar.metric("예상 가용 병상 비율", f"{predicted_ratio:.0%}")
    st.sidebar.markdown(f"**예측 혼잡도:** **:{predicted_level}**")

# ------------------------------
# 11. 추가 정보
# ------------------------------
st.markdown("---")
st.caption("※ 혼잡도는 K-means 군집 및 의료인/병상 비율 기반으로 산출되었습니다. 가용 병상은 실시간 API 미연동 시 표시되지 않습니다.")

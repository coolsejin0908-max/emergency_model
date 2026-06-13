import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import random
from datetime import datetime

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
    # 가용 병상 = 전체 병상 * (0.2 ~ 0.9) 무작위 + 시간대 영향
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
# 3. 사용자 위치 획득 (수동 + 자동 fallback)
# ------------------------------
st.sidebar.header("📍 사용자 위치 설정")
use_auto_location = st.sidebar.checkbox("내 위치 자동 감지 (브라우저 허용 필요)", value=False)

if use_auto_location:
    # HTML/JS로 위치 가져오기 (streamlit의 임시 방법)
    location_html = """
    <script>
    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const output = document.getElementById('geo_output');
            output.value = `${lat},${lon}`;
            output.dispatchEvent(new Event('input'));
        },
        (error) => {
            document.getElementById('geo_output').value = "37.5665,126.9780";  // 서울시청 default
        }
    );
    </script>
    <input type="text" id="geo_output" style="display:none">
    """
    st.components.v1.html(location_html, height=0)
    # 실제로는 session_state에 저장하기 위해 별도 버튼 사용 권장
    # 간단히: st.text_input으로 수동 입력 유도
    st.sidebar.warning("자동 위치는 실패 시 수동으로 입력하세요.")
    user_lat = st.sidebar.number_input("위도 (lat)", value=37.5665, format="%.6f")
    user_lon = st.sidebar.number_input("경도 (lon)", value=126.9780, format="%.6f")
else:
    user_lat = st.sidebar.number_input("위도 (lat)", value=37.5665, format="%.6f", help="예: 37.5665")
    user_lon = st.sidebar.number_input("경도 (lon)", value=126.9780, format="%.6f", help="예: 126.9780")

# 현재 위치 표시
st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")

# ------------------------------
# 4. 부상 유형 선택 및 거리 필터
# ------------------------------
st.sidebar.header("🏥 추천 조건")
injury_type = st.sidebar.selectbox("부상/증상 유형", injury_options)
radius_km = st.sidebar.slider("검색 반경 (km)", min_value=1, max_value=20, value=10, step=1)

# 병원까지의 거리 계산 및 필터링
def calculate_distance(row, user_lat, user_lon):
    hosp_coord = (row["lat"], row["lon"])
    user_coord = (user_lat, user_lon)
    return geodesic(user_coord, hosp_coord).kilometers

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
    # 혼잡도 점수: 여유(2), 보통(1), 혼잡(0)
    congestion_score = {"여유": 2, "보통": 1, "혼잡": 0}[row["congestion_text"]]
    # 가까울수록 가산 (거리 역수)
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
st.sidebar.header("🔬 전문가 모드 (혼잡도 예측)")
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
# 8. 추가 정보 (실시간 병상 시뮬레이션 설명)
# ------------------------------
st.markdown("---")
st.caption("※ 가용 병상 및 혼잡도는 현재 시간대 기반으로 시뮬레이션되었습니다. 실제 데이터는 병원별 API 연동 시 정확해집니다.")

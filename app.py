import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import datetime
import json
import requests

# ------------------------------
# 0. M 좌표 (TM) -> 위도/경도 변환 (예시: 국토지리정보원 API 또는 공식 변환)
# 여기서는 간단한 TM to WGS84 샘플 함수 (실제로는 변환 파라미터 필요)
# ------------------------------
def tm_to_wgs84(tm_x, tm_y):
    """
    가상의 변환 함수 (실제로는 epsg:5174(중부원점) -> epsg:4326 변환 라이브러리 필요)
    예시로 입력값에 따라 단순히 x,y에 offset을 더해 위/경도 근사값 반환
    실제 사용 시 pyproj 또는 변환 API를 연동하세요.
    """
    # 임의의 변환: 서울 기준 TM 좌표(약 200000, 500000) -> 위도 37.56, 경도 126.97
    base_tm_x = 200000
    base_tm_y = 500000
    base_lat = 37.5665
    base_lon = 126.9780
    scale = 0.000001  # 1m -> 약 0.000001도 (근사)
    lat = base_lat + (tm_y - base_tm_y) * scale
    lon = base_lon + (tm_x - base_tm_x) * scale
    return lat, lon

# ------------------------------
# 1. 페이지 설정
# ------------------------------
st.set_page_config(page_title="응급실 혼잡도 대시보드", layout="wide")
st.title("🏥 응급실 호환도 예측 대시보드")
st.markdown("---")

# ------------------------------
# 2. 데이터 로드 (M 좌표 포함 예시)
# ------------------------------
@st.cache_data
def load_hospital_data_with_m():
    # 실제 M 좌표를 포함한 샘플 데이터 (국토부 공공데이터 등에서 획득 가정)
    data = {
        "name": ["서울대병원", "세브란스병원", "삼성서울병원", "서울아산병원", "강남세브란스"],
        "tm_x": [198000, 195500, 203000, 201000, 206000],  # 가상의 TM X 좌표
        "tm_y": [542000, 541500, 540000, 540500, 539000],
        "beds": [120, 90, 110, 95, 70],
        "doctors": [150, 120, 140, 130, 90],
        "specialties": ["외상,심장,신경", "외상,심장", "외상,심장,정형", "심장,신경,소아", "외상,정형"],
    }
    df = pd.DataFrame(data)
    
    # M 좌표 -> 위도/경도 변환
    df[["lat", "lon"]] = df.apply(lambda row: pd.Series(tm_to_wgs84(row["tm_x"], row["tm_y"])), axis=1)
    
    # 전체 병원 수 표시용 (실제로는 1407개처럼 많아야 함 - 여기서는 5개지만 확장 가능)
    # 예시로 데이터프레임을 1407개로 늘리려면 반복 또는 실제 CSV 로드 필요.
    # 데모에서는 "유효한 병원 수: 5 (데모)" 로 표시
    return df

hospitals_df = load_hospital_data_with_m()
st.sidebar.metric("📊 유효한 병원 수", len(hospitals_df))

# ------------------------------
# 3. 사용자 위치
# ------------------------------
st.sidebar.header("📍 내 위치")
user_lat = st.sidebar.number_input("위도", value=37.5665, format="%.6f")
user_lon = st.sidebar.number_input("경도", value=126.9780, format="%.6f")

# ------------------------------
# 4. 모드 선택 (간편 모드 vs 전문가 모드)
# ------------------------------
mode = st.sidebar.radio("입력 방식", ["간편 모드 (질문)", "전문가 모드 (슬라이더)"])

if mode == "간편 모드 (질문)":
    st.sidebar.subheader("❓ 간편 질문")
    symptom = st.sidebar.selectbox("어떤 증상/부상이 있나요?", 
                                    ["골절", "심장 통증", "호흡곤란", "뇌졸중 의심", "교통사고 외상"])
    # 질문에 따라 필요한 전문과목 매핑
    symptom_to_specialty = {
        "골절": "정형", "심장 통증": "심장", "호흡곤란": "심장",
        "뇌졸중 의심": "신경", "교통사고 외상": "외상"
    }
    required_specialty = symptom_to_specialty[symptom]
    # 간편 모드에서는 병원 규모/의사 수 등을 자동 추천
    hospital_size_pref = st.sidebar.selectbox("선호하는 병원 규모", ["작은 병원", "중간 병원", "큰 병원"])
    # 전문가 모드용 변수 기본값
    expert_beds = 150
    expert_staff = 80
else:
    st.sidebar.subheader("🔬 전문가 모드")
    expert_beds = st.sidebar.number_input("병상 수 (개)", 20, 300, 150)
    expert_staff = st.sidebar.number_input("의료인 수 (명)", 10, 200, 80)
    hospital_size_pref = st.sidebar.selectbox("병원 규모 필터", ["모든 규모", "작은 병원 (<=30병상)", "중간 병원 (31~80)", "큰 병원 (>=81)"])
    required_specialty = st.sidebar.text_input("필수 전문과목 (예: 심장)", "외상")

# ------------------------------
# 5. 내 병상/의료인 수 예측 (사용자 자신의 응급실이 있다고 가정)
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 나의 응급실 예측")
my_beds = st.sidebar.number_input("내 병상 수 (개)", 5, 200, 20)
my_staff = st.sidebar.number_input("내 의료인 수 (명)", 1, 100, 2)
# 간단 예측 모델: 가용 병상률 = (my_staff / (my_beds * 0.5)) 클리핑
predicted_avail_rate = min(0.9, max(0.05, (my_staff / 100) * (my_beds / 50)))
predicted_congestion = "여유" if predicted_avail_rate > 0.5 else ("보통" if predicted_avail_rate > 0.2 else "혼잡")
st.sidebar.metric("나의 응급실 예상 혼잡도", predicted_congestion, f"가용률 {predicted_avail_rate:.0%}")

# ------------------------------
# 6. 거리 계산 및 필터링 (Haversine)
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlam = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

radius_km = st.sidebar.slider("검색 반경 (km)", 1, 30, 10)
hospitals_df["distance"] = hospitals_df.apply(lambda row: haversine(user_lat, user_lon, row["lat"], row["lon"]), axis=1)
nearby = hospitals_df[hospitals_df["distance"] <= radius_km].copy()

if nearby.empty:
    st.warning(f"반경 {radius_km}km 내 병원 없음")
    st.stop()

# 병원 규모 필터링
if "작은 병원" in hospital_size_pref:
    nearby = nearby[nearby["beds"] <= 30]
elif "중간 병원" in hospital_size_pref:
    nearby = nearby[(nearby["beds"] > 30) & (nearby["beds"] <= 80)]
elif "큰 병원" in hospital_size_pref:
    nearby = nearby[nearby["beds"] >= 81]

# 전문과목 적합성 점수
def match_specialty(row, required):
    spec_list = row["specialties"].split(",")
    return 1 if required in spec_list else 0

nearby["specialty_match"] = nearby.apply(lambda row: match_specialty(row, required_specialty), axis=1)
# 혼잡도 (가상: 병상 대비 의사 수 + 재원률 가정)
nearby["congestion"] = nearby.apply(lambda row: "혼잡" if row["doctors"]/row["beds"] < 1.2 else ("보통" if row["doctors"]/row["beds"] < 2.0 else "여유"), axis=1)
color_map = {"혼잡":"red", "보통":"orange", "여유":"green"}
nearby["marker_color"] = nearby["congestion"].map(color_map)

# 추천 점수 계산
nearby["score"] = nearby["specialty_match"] * 10 + (1 / (nearby["distance"]+0.1)) + ({"여유":3, "보통":1, "혼잡":0}[nearby["congestion"]])

nearby = nearby.sort_values("score", ascending=False)

# ------------------------------
# 7. 지도 표시 (히트맵 토글)
# ------------------------------
st.subheader("🗺️ 응급의료시설 현황")
show_heatmap = st.checkbox("히트맵으로 보기 (밀집도)", value=False)

map_center = [user_lat, user_lon]
m = folium.Map(location=map_center, zoom_start=12)

# 사용자 마커
folium.Marker([user_lat, user_lon], popup="내 위치", icon=folium.Icon(color="blue")).add_to(m)

if show_heatmap:
    # 히트맵 데이터 (위도, 경도, 가중치 = 혼잡도 점수)
    heat_data = [[row["lat"], row["lon"], 1 if row["congestion"]=="혼잡" else 0.5] for idx, row in nearby.iterrows()]
    HeatMap(heat_data).add_to(m)
else:
    for idx, row in nearby.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=f"{row['name']}<br>병상:{row['beds']} 의사:{row['doctors']}<br>혼잡도:{row['congestion']}",
            icon=folium.Icon(color=row["marker_color"])
        ).add_to(m)

folium_static(m, width=1000, height=500)

# ------------------------------
# 8. 추천 병원 테이블
# ------------------------------
st.subheader("🏥 추천 병원 목록")
st.dataframe(nearby[["name", "distance", "beds", "doctors", "congestion", "specialties", "score"]].style.format({"distance":"{:.1f}km", "score":"{:.1f}"}))

# ------------------------------
# 9. 전문가 모드 추가 예측 (슬라이더)
# ------------------------------
if mode == "전문가 모드 (슬라이더)":
    st.markdown("---")
    st.subheader("🔬 전문가 혼잡도 예측 (슬라이더 기반)")
    col1, col2 = st.columns(2)
    with col1:
        expert_beds = st.number_input("병상 수 (전문가 예측용)", 20, 500, expert_beds)
        expert_rooms = st.number_input("입원실 수", 10, 200, 45)
    with col2:
        expert_doctors = st.number_input("의사 수", 5, 200, expert_staff)
        expert_nurses = st.number_input("간호사 수", 10, 300, 100)
    
    # 예측 공식 (간단한 가중치)
    predicted_avail = min(0.95, (expert_doctors+expert_nurses) / (expert_beds * 2.5))
    predicted_level = "여유" if predicted_avail > 0.6 else ("보통" if predicted_avail > 0.3 else "혼잡")
    st.success(f"예상 혼잡도: **{predicted_level}** (가용 병상 비율 예측: {predicted_avail:.0%})")

st.caption("※ M 좌표 변환은 샘플 함수로 실제 데이터 연동 시 pyproj 라이브러리 권장")

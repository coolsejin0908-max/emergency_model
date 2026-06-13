import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="응급실 응급 추천", layout="wide")

st.title("🏥 응급실 혼잡도 및 부상 기반 추천")
st.markdown("---")

# ---------- CSV 로드 및 컬럼 매핑 (자동 감지) ----------
@st.cache_data
def load_data():
    df = pd.read_csv("emergency_hospitals.csv")
    st.write("### 🔍 CSV 로드 결과")
    st.write(f"전체 행 수: {len(df)}")
    st.write("컬럼 목록:", list(df.columns))
    
    # 컬럼 이름 매핑 (대소문자 무시, 유사한 이름 허용)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if '사업장' in col_lower or '병원명' in col_lower or 'name' in col_lower:
            col_map['name'] = col
        elif 'x' in col_lower or '경도' in col_lower or 'lon' in col_lower:
            col_map['lon'] = col
        elif 'y' in col_lower or '위도' in col_lower or 'lat' in col_lower:
            col_map['lat'] = col
        elif '혼잡' in col_lower:
            col_map['congestion'] = col
        elif '병상' in col_lower:
            col_map['beds'] = col
    
    # 필수 컬럼 체크
    if 'name' not in col_map or 'lat' not in col_map or 'lon' not in col_map:
        st.error("필수 컬럼(병원명, 위도, 경도)을 찾을 수 없습니다. CSV 파일을 확인하세요.")
        st.stop()
    
    # 필요한 컬럼만 추출하고 이름 통일
    df = df.rename(columns={
        col_map['name']: '사업장명',
        col_map['lat']: '좌표정보(Y)',
        col_map['lon']: '좌표정보(X)'
    })
    if 'congestion' in col_map:
        df = df.rename(columns={col_map['congestion']: '혼잡도'})
    else:
        df['혼잡도'] = "보통"  # 기본값
    
    # 좌표 숫자 변환
    df['좌표정보(X)'] = pd.to_numeric(df['좌표정보(X)'], errors='coerce')
    df['좌표정보(Y)'] = pd.to_numeric(df['좌표정보(Y)'], errors='coerce')
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    
    st.write(f"좌표 유효한 행 수: {len(df)}")
    if len(df) > 0:
        st.write("좌표 예시:", df[['사업장명', '좌표정보(X)', '좌표정보(Y)']].head(3))
    else:
        st.error("좌표 값이 모두 유효하지 않습니다. CSV의 위도/경도 열에 숫자가 있는지 확인하세요.")
    return df

df_hosp = load_data()

if df_hosp.empty:
    st.error("병원 데이터를 불러올 수 없어 앱을 종료합니다.")
    st.stop()

# ---------- 사용자 위치 ----------
st.sidebar.header("📍 내 위치")
loc = streamlit_geolocation()
if loc and loc.get('latitude') is not None:
    user_lat = loc['latitude']
    user_lon = loc['longitude']
    st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")
else:
    user_lat = st.sidebar.number_input("위도", value=37.5665, format="%.6f")
    user_lon = st.sidebar.number_input("경도", value=126.9780, format="%.6f")
    st.sidebar.info("기본 위치(서울시청)")

radius = st.sidebar.slider("검색 반경 (km)", 1, 200, 50)

# ---------- 거리 계산 ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

df_hosp['distance'] = df_hosp.apply(
    lambda row: haversine(user_lat, user_lon, row['좌표정보(Y)'], row['좌표정보(X)']), axis=1
)
near = df_hosp[df_hosp['distance'] <= radius].copy()
st.sidebar.metric("반경 내 병원 수", len(near))

# ---------- 메인 지도 ----------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("🗺️ 응급실 지도")
    if near.empty:
        st.warning(f"반경 {radius}km 내 응급실이 없습니다. 반경을 늘리거나 위치를 확인하세요.")
    else:
        m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
        folium.TileLayer('OpenStreetMap').add_to(m)
        # 사용자 마커
        folium.Marker([user_lat, user_lon], popup="내 위치", icon=folium.Icon(color='red')).add_to(m)
        # 병원 마커
        for _, row in near.iterrows():
            color = {'혼잡':'red', '보통':'orange', '여유':'green'}.get(row['혼잡도'], 'blue')
            folium.Marker(
                [row['좌표정보(Y)'], row['좌표정보(X)']],
                popup=f"{row['사업장명']}<br>거리: {row['distance']:.1f}km<br>혼잡도: {row['혼잡도']}",
                icon=folium.Icon(color=color)
            ).add_to(m)
        st_folium(m, width=700, height=500)

with col2:
    st.subheader("📋 추천 병원")
    if not near.empty:
        # 간단 추천: 거리 가까운 순 + 혼잡도 가중치
        near['score'] = near.apply(lambda r: 1/(r['distance']+0.1) + (2 if r['혼잡도']=='여유' else 1 if r['혼잡도']=='보통' else 0), axis=1)
        top = near.sort_values('score', ascending=False).iloc[0]
        st.success(f"### 최우선: {top['사업장명']}")
        st.write(f"거리: {top['distance']:.1f} km")
        st.write(f"혼잡도: {top['혼잡도']}")
    else:
        st.info("반경 내 병원 없음")

st.markdown("---")
st.subheader("📋 반경 내 응급실 목록")
if not near.empty:
    st.dataframe(near[['사업장명', 'distance', '혼잡도']].head(20).style.format({'distance': '{:.1f}'}))
else:
    st.info("없음")

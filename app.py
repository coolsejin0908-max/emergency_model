import streamlit as st
import pandas as pd
import math
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="응급실 혼잡도 추천", layout="wide")
st.title("🏥 응급실 혼잡도 및 부상 기반 추천")

# ---------- 데이터 로드 ----------
@st.cache_data
def load_data():
    df = pd.read_csv("emergency_hospitals.csv")
    # 컬럼명이 실제 CSV와 일치하는지 확인 (사업장명, 좌표정보(X), 좌표정보(Y), 혼잡도)
    required = ['사업장명', '좌표정보(X)', '좌표정보(Y)']
    for col in required:
        if col not in df.columns:
            st.error(f"CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
            st.stop()
    # 좌표를 숫자로 변환
    df['좌표정보(X)'] = pd.to_numeric(df['좌표정보(X)'], errors='coerce')
    df['좌표정보(Y)'] = pd.to_numeric(df['좌표정보(Y)'], errors='coerce')
    df = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)'])
    # 혼잡도 컬럼이 없으면 기본값 추가
    if '혼잡도' not in df.columns:
        df['혼잡도'] = '보통'
    return df

df = load_data()
st.sidebar.write(f"✅ 전체 병원 수: {len(df)}")

# ---------- 사용자 위치 ----------
st.sidebar.header("📍 내 위치")
location = streamlit_geolocation()
if location and location.get('latitude'):
    user_lat = location['latitude']
    user_lon = location['longitude']
    st.sidebar.success(f"현재 위치: {user_lat:.4f}, {user_lon:.4f}")
else:
    user_lat = st.sidebar.number_input("위도 (수동)", value=37.5665, format="%.6f")
    user_lon = st.sidebar.number_input("경도 (수동)", value=126.9780, format="%.6f")
    st.sidebar.info("기본 위치: 서울시청")

radius = st.sidebar.slider("검색 반경 (km)", 1, 200, 50)

# ---------- 거리 계산 ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

df['distance'] = df.apply(lambda row: haversine(user_lat, user_lon, row['좌표정보(Y)'], row['좌표정보(X)']), axis=1)
near = df[df['distance'] <= radius].copy()
st.sidebar.metric("반경 내 병원 수", len(near))

# ---------- 지도 및 추천 ----------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🗺️ 응급실 지도")
    if near.empty:
        st.warning(f"반경 {radius}km 내 응급실이 없습니다. 반경을 늘리거나 위치를 확인하세요.")
    else:
        m = folium.Map(location=[user_lat, user_lon], zoom_start=12)
        folium.Marker([user_lat, user_lon], popup="내 위치", icon=folium.Icon(color='red')).add_to(m)
        for _, row in near.iterrows():
            color = {'혼잡': 'red', '보통': 'orange', '여유': 'green'}.get(row['혼잡도'], 'blue')
            folium.Marker(
                [row['좌표정보(Y)'], row['좌표정보(X)']],
                popup=f"{row['사업장명']}<br>거리: {row['distance']:.1f}km<br>혼잡도: {row['혼잡도']}",
                icon=folium.Icon(color=color)
            ).add_to(m)
        st_folium(m, width=700, height=500)

with col2:
    st.subheader("📋 추천 순위")
    if not near.empty:
        # 간단 추천 점수: 거리 가까울수록, 혼잡도 낮을수록 높음
        near['score'] = near.apply(lambda r: 1/(r['distance']+0.1) + (2 if r['혼잡도']=='여유' else 1 if r['혼잡도']=='보통' else 0), axis=1)
        top = near.sort_values('score', ascending=False).head(5)
        for i, row in top.iterrows():
            st.write(f"**{row['사업장명']}**  \n거리 {row['distance']:.1f}km, {row['혼잡도']}")
    else:
        st.info("반경 내 병원 없음")

st.markdown("---")
st.subheader("📋 반경 내 응급실 목록")
if not near.empty:
    st.dataframe(near[['사업장명', 'distance', '혼잡도']].head(20).style.format({'distance': '{:.1f}'}))

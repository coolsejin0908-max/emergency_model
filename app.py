import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(page_title="응급실 혼잡도 지도", layout="wide")

# 1. CSV 파일 로드 (혼잡도 컬럼이 미리 계산되어 있다고 가정)
#    만약 없다면, 여기서 df_new를 읽고 모델로 혼잡도를 계산한 후 저장하는 로직 필요
df = pd.read_csv("emergency_hospitals.csv")  # 미리 준비된 파일

# 2. 지도 생성
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

# 3. CSV 데이터로 마커 추가
for _, row in df.iterrows():
    # 혼잡도에 따른 색상
    congestion = row.get('혼잡도', '정보 없음')
    if congestion == '혼잡':
        color = 'red'
    elif congestion == '보통':
        color = 'orange'
    else:
        color = 'green'
    
    folium.Marker(
        location=[row['좌표정보(Y)'], row['좌표정보(X)']],  # 위도, 경도 순서
        popup=f"{row['사업장명']}<br>혼잡도: {congestion}<br>병상수: {row['병상수']}",
        icon=folium.Icon(color=color, icon='plus', prefix='fa')
    ).add_to(m)

# 4. Streamlit에 표시
st.title("🚑 응급실 혼잡도 현황")
st_folium(m, width=1000, height=600)

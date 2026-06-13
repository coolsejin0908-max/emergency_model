# 1. 필요 라이브러리 설치 및 임포트
!pip install folium utm -q

import folium
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# 2. 데이터 로드
df = pd.read_csv('emergency_hospitals.csv')

# 컬럼명 확인 (필요시 수정)
print("컬럼 목록:", df.columns.tolist())

# --- 컬럼 매핑 (CSV 구조에 맞게) ---
name_col = '사업장명'
lat_col = '좌표정보(Y)'   # 위도
lon_col = '좌표정보(X)'   # 경도

# --- 좌표 숫자 변환 및 결측치 제거 ---
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
df = df.dropna(subset=[lat_col, lon_col])

# --- UTM → 위도/경도 변환 (필요시) ---
if len(df) > 0:
    sample_lon = df[lon_col].iloc[0]
    if abs(sample_lon) > 180 or sample_lon > 100000:
        try:
            import utm
            def convert_utm(row):
                try:
                    lat, lon = utm.to_latlon(row[lon_col], row[lat_col], 52, 'N')
                    return lat, lon
                except:
                    return None, None
            df['위도'], df['경도'] = zip(*df.apply(convert_utm, axis=1))
            lat_col, lon_col = '위도', '경도'
            df = df.dropna(subset=[lat_col, lon_col])
            print("UTM → WGS84 변환 완료")
        except ImportError:
            print("utm 모듈 없음. 변환 생략")
    else:
        print("이미 WGS84 좌표로 간주")
else:
    print("⚠️ 데이터가 없습니다.")

# --- 좌표 유효성 검사 ---
df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]
print(f"유효한 병원 수: {len(df)}")

if len(df) == 0:
    print("⚠️ 표시할 수 있는 병원이 없습니다.")
else:
    # --- 사용자 위치 입력 (기본값: 서울 시청) ---
    print("\n📍 지도 중심 좌표를 입력하세요 (위도, 경도)")
    user_lat = float(input("위도 (default 37.5665): ") or "37.5665")
    user_lon = float(input("경도 (default 126.9780): ") or "126.9780")

    # --- 지도 생성 (사용자 위치 중심) ---
    m = folium.Map(location=[user_lat, user_lon], zoom_start=12)

    # 사용자 위치 마커 (빨간색, 아이콘)
    folium.Marker(
        location=[user_lat, user_lon],
        popup="내 위치",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)

    # 병원 마커 (파란색 원)
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            popup=row.get(name_col, '병원'),
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.5
        ).add_to(m)

    # 지도 출력
    display(m)

    # HTML 파일로 저장하려면 주석 해제
    # m.save('hospital_map_with_my_location.html')
    # print("지도가 hospital_map_with_my_location.html 로 저장되었습니다.")

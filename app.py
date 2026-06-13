import folium
import pandas as pd
import numpy as np
from folium.plugins import LocateControl   # 위치 권한 플러그인

# ----------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------

# CSV 파일 읽기 (실제 파일명에 맞게 수정)
df = pd.read_csv('emergency_hospitals.csv')

# 컬럼명 확인 (디버깅용)
print("컬럼 목록:", df.columns.tolist())

# --- 컬럼 매핑 (CSV 구조에 맞게 수정) ---
name_col = '사업장명'       # 병원 이름 컬럼
lat_col = '좌표정보(Y)'     # 위도 컬럼 (Latitude)
lon_col = '좌표정보(X)'     # 경도 컬럼 (Longitude)

# 좌표 컬럼을 숫자로 변환, 오류 시 NaN 처리
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

# 결측치 제거
df = df.dropna(subset=[lat_col, lon_col])

# --- UTM → 위도/경도 변환 (필요시) ---
# UTM 좌표는 보통 6자리 이상의 숫자이며 경도 범위(-180~180)를 벗어남
if len(df) > 0:
    sample_lon = df[lon_col].iloc[0]
    if abs(sample_lon) > 180 or sample_lon > 100000:
        try:
            import utm
            def convert_utm(row):
                try:
                    # UTM to Lat/Lon (zone 52, northern hemisphere)
                    lat, lon = utm.to_latlon(row[lon_col], row[lat_col], 52, 'N')
                    return lat, lon
                except:
                    return None, None
            df['위도'], df['경도'] = zip(*df.apply(convert_utm, axis=1))
            lat_col, lon_col = '위도', '경도'
            df = df.dropna(subset=[lat_col, lon_col])
            print("UTM → WGS84 변환 완료")
        except ImportError:
            print("⚠️ utm 모듈이 없습니다. 'pip install utm'을 실행하세요.")
            df = df.head(0)  # 변환 불가 → 빈 데이터

# --- 좌표 유효성 검사 (위도 -90~90, 경도 -180~180) ---
df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]

if len(df) == 0:
    print("⚠️ 유효한 좌표 데이터가 없습니다. CSV 파일과 컬럼명을 확인하세요.")
else:
    # ----------------------------
    # 2. 지도 생성 및 마커 추가
    # ----------------------------
    
    # 지도 중심을 병원 데이터의 평균 좌표로 설정
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # --- 병원 위치에 원형 마커 추가 (지진 데이터 스타일: 테두리만 있는 원) ---
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,                       # 원의 크기 (픽셀)
            popup=row.get(name_col, '병원'), # 클릭 시 표시될 이름
            color='blue',                   # 테두리 색상
            weight=3,                       # 테두리 두께
            fill=False,                     # 채우기 없음 (지진 스타일)
            opacity=1.0                     # 불투명도
        ).add_to(m)

    # --- 사용자 현재 위치 제어 버튼 추가 (위치 권한 기능) ---
    LocateControl(
        position='topleft',
        strings={'title': '내 위치 찾기'},
        locateOptions={'enableHighAccuracy': True}
    ).add_to(m)

    # (선택) 지도 클릭 시 좌표 표시 (디버깅용)
    m.add_child(folium.LatLngPopup())

    # ----------------------------
    # 3. 지도 출력 및 저장
    # ----------------------------
    
    # Jupyter Notebook/Google Colab에서 바로 표시
    display(m)
    
    # HTML 파일로 저장 (필요시 주석 해제)
    # m.save('hospital_map.html')

# 1. utm 라이브러리 설치
!pip install utm

import pandas as pd
import utm

# 2. CSV 파일 업로드 (emergency_hospitals.csv)
from google.colab import files
uploaded = files.upload()   # 파일 선택

# 3. CSV 로드
df = pd.read_csv('emergency_hospitals.csv')

# 4. UTM → 위도/경도 변환 함수 (zone 52N, 실패시 51N)
def convert_utm(x, y):
    for zone in [52, 51]:
        for hemisphere in ['N', 'S']:
            try:
                lat, lon = utm.to_latlon(x, y, zone, hemisphere)
                # 한국 위도/경도 범위 (대략 33~39, 124~132)
                if 33 <= lat <= 39 and 124 <= lon <= 132:
                    return lat, lon
            except:
                continue
    return None, None

# 5. 변환 적용
coords = df.apply(lambda row: convert_utm(row['좌표정보(X)'], row['좌표정보(Y)']), axis=1)
df['위도'], df['경도'] = zip(*coords)
df = df.dropna(subset=['위도', '경도'])

# 6. 기존 좌표 컬럼 대체
df['좌표정보(Y)'] = df['위도']
df['좌표정보(X)'] = df['경도']
df = df.drop(columns=['위도', '경도'])

# 7. 새 CSV 저장
df.to_csv('emergency_hospitals_wgs84.csv', index=False, encoding='utf-8-sig')
print(f"변환 성공! 유효한 병원 수: {len(df)}")
print(df[['사업장명', '좌표정보(X)', '좌표정보(Y)']].head())

# 8. 다운로드
files.download('emergency_hospitals_wgs84.csv')

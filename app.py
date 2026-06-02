# ========================
# 1. 라이브러리 임포트
# ========================
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ========================
# 2. 페이지 설정
# ========================
st.set_page_config(
    page_title="AI 응급실 대기시간 예측기",
    page_icon="🚑",
    layout="wide"
)

# ========================
# 3. API 서버 주소 (로컬 또는 배포된 주소)
# ========================
# 로컬 테스트 시: http://localhost:8000
# Streamlit Cloud 등 배포 시에는 실제 API 서버 URL로 변경
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# ========================
# 4. 타이틀 및 설명
# ========================
st.title("🚑 AI 기반 응급실 대기시간 예측 & 추천 서비스")
st.markdown("""
이 서비스는 **공공 응급의료 데이터**를 기반으로  
현재 환자 수, 가용 병상 수, 중증도를 고려하여 **가장 빠른 응급실을 추천**합니다.
""")

# ========================
# 5. 사이드바 입력
# ========================
with st.sidebar:
    st.header("👤 환자 정보")
    severity_map = {
        "경증 (1)": 1,
        "보통 (2)": 2,
        "중증 (3)": 3,
        "위중 (4)": 4,
        "심각 (5)": 5
    }
    severity_text = st.selectbox("중증도", list(severity_map.keys()), index=2)
    severity_level = severity_map[severity_text]
    
    location = st.text_input("현재 위치 (예: 서울특별시 강남구)", "서울특별시 강남구")
    
    st.markdown("---")
    st.caption("※ 예측은 AI 모델에 의한 추정치이며, 실제 대기시간과 다를 수 있습니다.")

# ========================
# 6. 메인 버튼 및 결과 표시
# ========================
if st.button("🔍 가장 빠른 응급실 찾기", type="primary", use_container_width=True):
    with st.spinner("공공데이터를 불러오고 AI가 예측하는 중입니다..."):
        try:
            # 1) 주변 응급실 목록 가져오기 (API 서버가 알아서 수집하도록 요청)
            #    이때 er_list를 비워두면 서버 내부에서 공공API 호출
            request_body = {
                "user_location": location,
                "severity_level": severity_level,
                "er_list": None   # 서버에서 자체 수집하도록 함
            }
            
            # 2) 추천 API 호출
            response = requests.post(
                f"{API_BASE_URL}/recommend",
                json=request_body,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                best = data["best_er"]
                recommendations = data["recommendations"]
                
                # 3) 성공 메시지 및 최적 추천 표시
                st.success(f"🏥 **최적 추천 응급실:** {best['name']}  \n"
                           f"⏱️ 예상 대기시간: **{best['predicted_waiting_minutes']}분**")
                
                # 4) 전체 결과 테이블
                st.subheader("📋 응급실별 예상 대기시간")
                df = pd.DataFrame(recommendations)
                # 컬럼명 예쁘게 변경
                df_display = df.rename(columns={
                    "name": "응급실 이름",
                    "predicted_waiting_minutes": "예상 대기시간(분)",
                    "current_patients": "현재 환자 수",
                    "available_beds": "가용 병상 수"
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # 5) 시각화 (막대 그래프)
                st.subheader("📊 응급실별 대기시간 비교")
                fig = px.bar(
                    df,
                    x="name",
                    y="predicted_waiting_minutes",
                    title="응급실별 예상 대기시간",
                    labels={"name": "응급실", "predicted_waiting_minutes": "대기시간 (분)"},
                    color="predicted_waiting_minutes",
                    color_continuous_scale="Reds",
                    text="predicted_waiting_minutes"
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                
                # 6) 추가 정보: 현재 시간 기준 예측
                st.caption(f"🕒 예측 기준 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            else:
                st.error(f"❌ API 호출 실패 (상태 코드: {response.status_code})\n"
                         f"응답 내용: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.\n"
                     f"현재 설정된 API 주소: `{API_BASE_URL}`")
        except Exception as e:
            st.error(f"⚠️ 오류 발생: {str(e)}")

# ========================
# 7. 푸터 (추가 설명)
# ========================
st.markdown("---")
st.markdown("""
**※ 데이터 출처:** 공공데이터포털 응급의료기관 정보 (예시)  
**※ 모델:** RandomForestRegressor 기반 대기시간 회귀 예측  
**※ 문의:** 프로젝트 탐구 과제용 데모입니다.
""")

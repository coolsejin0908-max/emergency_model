import streamlit as st
from model import predict_emergency_congestion

st.set_page_config(page_title="응급실 혼잡도 예측", layout="centered")
st.title("🏥 응급실 혼잡도 예측 시스템")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        bed = st.number_input("병상수", min_value=1.0, step=1.0)
        room = st.number_input("입원실수", min_value=1.0, step=1.0)
        doctor = st.number_input("의료인수", min_value=1.0, step=1.0)
    with col2:
        cluster = st.selectbox("군집 번호", [0, 1, 2], format_func=lambda x: {0:"중소형",1:"중대형",2:"초대형"}[x])
        med_type = st.selectbox("의료기관종별명", ["종합병원", "병원"])
    
    submitted = st.form_submit_button("혼잡도 예측")

if submitted:
    with st.spinner("예측 중..."):
        pred, prob = predict_emergency_congestion(bed, room, doctor, cluster, med_type)
    
    st.success(f"### 예측 결과: **{pred}**")
    
    # 확률을 게이지 형태로 표시
    st.subheader("클래스별 확률")
    for label, p in prob.items():
        st.progress(p, text=f"{label}: {p:.2%}")

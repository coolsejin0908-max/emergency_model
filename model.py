import joblib
import pandas as pd
import os

# 모델 파일 경로 (GitHub에서는 상대 경로로 충분)
BASE_DIR = os.path.dirname(__file__)

rf = joblib.load(os.path.join(BASE_DIR, 'random_forest_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_ml.pkl'))
le_type = joblib.load(os.path.join(BASE_DIR, 'le_medical_type.pkl'))
le_scale = joblib.load(os.path.join(BASE_DIR, 'le_bed_scale.pkl'))

def predict_emergency_congestion(bed, room, doctor, cluster, med_type):
    """
    bed : 병상수
    room : 입원실수
    doctor : 의료인수
    cluster : 군집 번호 (0,1,2)
    med_type : '종합병원' 또는 '병원'
    """
    # 파생 변수 계산
    occupancy = room / bed
    doctor_per_room = room / doctor
    is_general = 1 if med_type == '종합병원' else 0
    
    # 병상 규모 결정
    if bed <= 100:
        bed_scale = '소형'
    elif bed <= 300:
        bed_scale = '중형'
    else:
        bed_scale = '대형'
    
    # 인코딩
    med_enc = le_type.transform([med_type])[0]
    scale_enc = le_scale.transform([bed_scale])[0]
    
    # 입력 데이터프레임 (특성 순서 중요!)
    input_df = pd.DataFrame([[
        bed, room, doctor, cluster,
        doctor / bed, occupancy, doctor_per_room,
        is_general, scale_enc
    ]], columns=[
        '병상수', '입원실수', '의료인수', 'cluster',
        '의료인_병상_비율', '입원실_점유율', '의료인_당_입원실',
        '종합병원_여부', '병상_규모_encoded'
    ])
    
    # 스케일링 및 예측
    input_scaled = scaler.transform(input_df)
    pred = rf.predict(input_scaled)[0]
    prob = rf.predict_proba(input_scaled)[0]
    prob_dict = dict(zip(rf.classes_, prob))
    
    return pred, prob_dict

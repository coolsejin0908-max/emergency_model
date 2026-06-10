from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 모델 및 전처리 도구 로드
rf = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler_ml.pkl')
le_type = joblib.load('le_medical_type.pkl')
le_scale = joblib.load('le_bed_scale.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 폼에서 입력 받기
    bed = float(request.form['bed'])
    room = float(request.form['room'])
    doctor = float(request.form['doctor'])
    cluster = int(request.form['cluster'])
    med_type = request.form['med_type']
    
    # 파생 변수 계산
    occupancy = room / bed
    doctor_per_room = room / doctor
    is_general = 1 if med_type == '종합병원' else 0
    # 병상 규모
    if bed <= 100:
        bed_scale = '소형'
    elif bed <= 300:
        bed_scale = '중형'
    else:
        bed_scale = '대형'
    
    # 인코딩
    med_enc = le_type.transform([med_type])[0]
    scale_enc = le_scale.transform([bed_scale])[0]
    
    # 특징 배열 생성 (순서 중요!)
    X = pd.DataFrame([[
        bed, room, doctor, cluster,
        doctor/bed, occupancy, doctor_per_room,
        is_general, scale_enc
    ]], columns=[
        '병상수', '입원실수', '의료인수', 'cluster',
        '의료인_병상_비율', '입원실_점유율', '의료인_당_입원실',
        '종합병원_여부', '병상_규모_encoded'
    ])
    
    X_scaled = scaler.transform(X)
    pred = rf.predict(X_scaled)[0]
    prob = rf.predict_proba(X_scaled)[0]
    prob_dict = dict(zip(rf.classes_, prob))
    
    return render_template('result.html', pred=pred, prob=prob_dict)

if __name__ == '__main__':
    app.run(debug=True)

import joblib
import pandas as pd
from datetime import datetime

class WaitingTimeModel:
    def __init__(self, model_path="er_waiting_model.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = ['current_patients', 'available_beds', 'severity_level', 'hour', 'dayofweek']

    def predict(self, current_patients, available_beds, severity_level, hour=None, dayofweek=None):
        if hour is None:
            hour = datetime.now().hour
        if dayofweek is None:
            dayofweek = datetime.now().weekday()

        input_df = pd.DataFrame([[
            current_patients, available_beds, severity_level, hour, dayofweek
        ]], columns=self.feature_names)
        pred = self.model.predict(input_df)[0]
        return round(float(pred), 1)

from datetime import datetime

import numpy
import pandas as pd
import onnxruntime as rt
import pickle

from sqlalchemy import select, create_engine, func

from data_models import PPGFeatures


class CFPrediction:
    def __init__(self):
        self.columns = ['bpm', 'lf_hf', 'ibi', 'hf_perc', 'vlf_perc', 'sdsd', 'sdnn', 'sd1_sd2',
                        'breathingrate', 'sd1', 'sd2', 'rmssd', 'pnn50', 's', 'lf', 'pnn20', 'hr_mad',
                        'p_total', 'vlf', 'hf', 'lf_perc']
        # Load the model
        sess = rt.InferenceSession("model.onnx")
        self.inferenceSession = sess

    def predict(self, engine, userID, date):
        query = f"SELECT * FROM ppg_features WHERE user_id = :user_id"
        df = pd.read_sql(query, engine, params={"user_id": userID})
        df_features = df[self.columns]

        input_name = self.inferenceSession.get_inputs()[0].name
        pred_onx = self.inferenceSession.run(None, {input_name: df_features.astype(numpy.float32).values})[0]

        df['fatigue'] = pred_onx
        df['finish_date'] = df['finish_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").date())
        df['finish_time'] = df['finish_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))
        df = df[df['finish_date'] == date]

        return df[['fatigue', 'finish_time']]

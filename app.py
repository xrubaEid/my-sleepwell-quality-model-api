from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SleepQualityInput(BaseModel):
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q6: str
    Q7: str

# تحميل النموذج المدرب و LabelEncoder
best_model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.post('/predict_sleep_quality')
def predict_sleep_quality(input_parameters: SleepQualityInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    answers = [
        input_dictionary['Q1'],
        input_dictionary['Q2'],
        input_dictionary['Q3'],
        input_dictionary['Q4'],
        input_dictionary['Q5'],
        input_dictionary['Q6'],
        input_dictionary['Q7']
    ]

    # تحويل المدخلات إلى نفس التنسيق المستخدم في التدريب
    input_data = {
        'Q1': [answers[0]],
        'Q2': [answers[1]],
        'Q3': [answers[2]],
        'Q4': [answers[3]],
        'Q5': [answers[4]],
        'Q6': [answers[5]],
        'Q7': [answers[6]]
    }

    input_df = pd.DataFrame(input_data)
    input_df = pd.get_dummies(input_df)

    # ضمان أن تكون الأعمدة في المدخلات الجديدة تتطابق مع الأعمدة المستخدمة في التدريب
    missing_cols = set(best_model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[best_model.feature_names_in_]

    # تنفيذ التنبؤ
    predicted_quality_encoded = best_model.predict(input_df)

    # تحويل التنبؤ إلى التصنيف الأصلي
    predicted_quality = label_encoder.inverse_transform(predicted_quality_encoded)

    return {'Predicted Sleep Quality': predicted_quality[0]}

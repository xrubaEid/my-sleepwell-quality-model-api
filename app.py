from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
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

class ModelInput(BaseModel):
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q6: str
    Q7: str

# تحميل النموذج المحفوظ
model = pickle.load(open('sleepwell_quality_trained_model.sav', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.post('/predict_sleep_quality')
def predict_sleep_quality(input_data: ModelInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # تحويل المدخلات إلى نفس التنسيق المستخدم في التدريب
    input_df = pd.get_dummies(input_df)
    
    # ضمان أن تكون الأعمدة في المدخلات الجديدة تتطابق مع الأعمدة المستخدمة في التدريب
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    # تنفيذ التنبؤ
    predicted_quality_encoded = model.predict(input_df)
    
    # تحويل التنبؤ إلى التصنيف الأصلي
    predicted_quality = label_encoder.inverse_transform(predicted_quality_encoded)
    
    return {"Predicted Sleep Quality": predicted_quality[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

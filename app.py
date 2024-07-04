from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import joblib

# Load model and label encoder
model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create FastAPI app instance
app = FastAPI()

# Define input structure using Pydantic
class SleepQualityInput(BaseModel):
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q6: str
    Q7: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sleep Quality API"}

# Prediction endpoint
@app.post("/predict_sleep_quality/")
def predict_sleep_quality(input_data: SleepQualityInput):
    try:
        # Prepare input data
        input_df = pd.DataFrame([input_data.dict()])
        input_df = pd.get_dummies(input_df)

        # Ensure input columns match model expectations
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        # Make prediction
        predicted_quality_encoded = model.predict(input_df)
        predicted_quality = label_encoder.inverse_transform(predicted_quality_encoded)

        return {"predicted_sleep_quality": predicted_quality[0]}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"ValueError: {str(ve)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

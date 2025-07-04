from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
try:
    model = joblib.load("xgboost_heart_attack_model.pkl")
    print("✅ Model loaded successfully...")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise e

app = FastAPI()

# Corrected schema
class HeartData(BaseModel):
    pressurehight: float
    pressurelow: float
    pulse_pressure: float
    bp_ratio: float
    glucose_troponin: float
    bp_glucose_ratio: float
    glucose_log: float
    impluse_log: float
    kcm_log: float
    troponin_log: float
    high_glucose_flag: int
    high_troponin: int
    high_kcm: int
    gender: int

@app.post("/predict")
def predict(data: HeartData):
    try:
        request_data = data.model_dump()
        print("📥 Incoming request data:", request_data)

        input_array = np.array([[
            request_data['pressurehight'], request_data['pressurelow'], request_data['pulse_pressure'],
            request_data['bp_ratio'], request_data['glucose_troponin'], request_data['bp_glucose_ratio'],
            request_data['glucose_log'], request_data['impluse_log'], request_data['kcm_log'],
            request_data['troponin_log'], request_data['high_glucose_flag'],
            request_data['high_troponin'], request_data['high_kcm'], request_data['gender']
        ]])

        prediction = model.predict(input_array)[0]
        label = "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"
        return {"status": "✅ Success", "prediction": label}

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))

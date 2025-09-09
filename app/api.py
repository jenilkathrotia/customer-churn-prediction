from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = Path("app/model.joblib")

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Initialize app
app = FastAPI(title="Customer Churn Prediction API")

# Request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump()])  # ✅ correct Pydantic v2 usage

    # Predict churn probability
    prob = model.predict_proba(df)[0, 1]

    return {"churn_probability": round(float(prob), 4)}


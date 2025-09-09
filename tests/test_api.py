import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.5,
    }

    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1


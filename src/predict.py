import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("app/model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict_single(customer_dict):
    """
    Predict churn probability for one customer.
    Example:
    customer_dict = {
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
    """
    model = load_model()
    df = pd.DataFrame([customer_dict])
    prob = model.predict_proba(df)[0, 1]
    return prob

if __name__ == "__main__":
    # Example customer
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

    prob = predict_single(sample_customer)
    print(f"Churn Probability: {prob:.2%}")

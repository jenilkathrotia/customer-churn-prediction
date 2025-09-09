import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("app/model.joblib")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# Input form
with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    df = pd.DataFrame([customer])
    prob = model.predict_proba(df)[0, 1]

    st.subheader("🔮 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")

    # Plot probability
    fig, ax = plt.subplots()
    ax.bar(["Stay", "Churn"], [1 - prob, prob], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)


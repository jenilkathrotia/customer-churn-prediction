# 📉 Customer Churn Prediction System  

[![CI](https://github.com/jenilkathrotia/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/jenilkathrotia/customer-churn-prediction/actions)
[![codecov](https://codecov.io/gh/jenilkathrotia/customer-churn-prediction/branch/main/graph/badge.svg?token=0ae8b17e-1f5c-4ef0-8c5a-2961eafc1337)](https://codecov.io/gh/jenilkathrotia/customer-churn-prediction)


## Overview
This project is a **machine learning application** that predicts whether a customer is likely to stop using a company’s product or service (**churn**).  

Churn is a critical problem in industries like **telecom, banking, SaaS, retail, e-commerce, and streaming services**.  
By predicting churn in advance, businesses can take preventive actions (e.g., discounts, personalized offers, improved support) to **retain customers and reduce revenue loss**.  

---

## Features
- **Data Preprocessing & Feature Engineering**  
  - Handles missing values, scales numerical features, and encodes categoricals  
  - Supports class imbalance techniques (SMOTE, class weights)

- **Machine Learning Models**  
  - Logistic Regression, Random Forest, XGBoost, Neural Nets (optional)  
  - Model selection based on **Precision-Recall AUC** and **ROC-AUC**

- **Explainability (XAI)**  
  - SHAP/LIME to explain predictions  
  - Identify top reasons for customer churn

- **Deployment**  
  - **Streamlit App** → interactive dashboard for predictions + explanations  
  - **FastAPI Service** → REST API endpoint for integration into CRM systems  
  - **Dockerized** → portable and deployable on AWS/GCP/Azure

---

##  Tech Stack
- **Languages:** Python 3.11  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, matplotlib, seaborn  
- **Web Frameworks:** Streamlit, FastAPI  
- **DevOps:** Docker, GitHub Actions (CI)  
- **Testing:** pytest  

---

## Dataset
- Example dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Target variable: `Churn` (`Yes` = 1, `No` = 0)  
- Features include demographics, services subscribed, billing/payment info, and usage behavior  

**Note:** Dataset is not included in the repo due to licensing. Please download from Kaggle and place in `data/`.

---

## Project Structure
customer-churn-prediction/
├── README.md # Project overview
├── benchmarks.md # Model benchmarks & results
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
├── data/ # Dataset (not included, add here)
├── notebooks/ # Jupyter notebooks (EDA, experiments)
├── src/ # ML pipeline code
│ ├── preprocess.py # Preprocessing & feature engineering
│ ├── train.py # Model training & evaluation
│ ├── explain.py # SHAP/LIME explainability
│ └── predict.py # Model prediction utilities
├── app/ # Deployment apps
│ ├── streamlit_app.py # Streamlit dashboard
│ └── api.py # FastAPI service
└── tests/ # Unit tests
├── test_model.py
└── test_api.py


---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/jenilkathrotiya/customer-churn-prediction.git
cd customer-churn-predictionimport joblib

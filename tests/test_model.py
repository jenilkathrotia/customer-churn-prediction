import os
import joblib

def test_model_exists():
    assert os.path.exists("app/model.joblib"), "❌ model.joblib not found"

def test_model_loads():
    model = joblib.load("app/model.joblib")
    assert model is not None, "❌ model could not be loaded"


import pytest
import joblib
from pathlib import Path

MODEL_PATH = Path("app/model.joblib")

def test_model_exists():
    assert MODEL_PATH.exists(), "Model file not found!"

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Model does not support predict_proba"


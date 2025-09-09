import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Paths
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = Path("app/model.joblib")

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Target = Churn (Yes=1, No=0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    # Convert TotalCharges to numeric (it has blanks)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    # Drop customerID (not useful for prediction)
    df = df.drop(columns=["customerID"])
    return df

def build_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return preprocessor

def evaluate(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    print(f"{name}: ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}")
    return pr

def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        ),
    }

    best_model = None
    best_score = -np.inf

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        pr_auc = evaluate(pipe, X_test, y_test, name)
        if pr_auc > best_score:
            best_score = pr_auc
            best_model = pipe

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()


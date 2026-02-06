import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CUSTOMER CHURN PREDICTION")
print("=" * 70)

# Load Data
print("\n[1/5] Loading Data...")
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    csv_path = 'data/'
else:
    csv_path = ''

csv_file = csv_path + csv_files[0]
df = pd.read_csv(csv_file)
print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Identify Target
print("\n[2/5] Finding Target Column...")
target_col = None
for col in ['Churn', 'churn', 'Churned', 'churned', 'target']:
    if col in df.columns:
        target_col = col
        break

print(f"   Target: '{target_col}'")
print(f"   Original values: {df[target_col].unique()}")

# Preprocess
print("\n[3/5] Preprocessing...")
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target to 0/1
le = LabelEncoder()
y = le.fit_transform(y)
print(f"   Encoded target: {le.classes_} -> {[0, 1]}")

id_cols = [col for col in X.columns if 'id' in col.lower()]
if id_cols:
    X = X.drop(columns=id_cols)

if X.isnull().sum().sum() > 0:
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"   Features: {X.shape}")

# Split
print("\n[4/5] Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("\n[5/5] Training Models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_f1 = 0
best_model = None
best_name = None
results = {}

for name, model in models.items():
    print(f"\n   {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")
    results[name] = {'Accuracy': acc, 'F1': f1}
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Save
os.makedirs('model_output', exist_ok=True)

artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': X.columns.tolist(),
    'label_encoder': le,
    'model_name': best_name
}

with open('model_output/model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

np.save('model_output/X_test.npy', X_test_scaled)
np.save('model_output/y_test.npy', y_test)

pd.DataFrame(results).T.to_csv('model_output/model_comparison.csv')

print("\n" + "=" * 70)
print(f"✅ BEST: {best_name}")
print(f"   Accuracy: {results[best_name]['Accuracy']:.4f}")
print(f"   F1-Score: {best_f1:.4f}")
print("\n📁 Saved to model_output/")
print("=" * 70)

# Show detailed report
y_pred_best = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

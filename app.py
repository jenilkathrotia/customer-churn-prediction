from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
with open('model_output/model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

MODEL = artifacts['model']
SCALER = artifacts['scaler']
FEATURES = artifacts['feature_columns']
LABEL_ENCODER = artifacts['label_encoder']

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'model': artifacts['model_name'],
        'accuracy': 0.7715,
        'features': len(FEATURES)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        # Ensure all features present
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[FEATURES]
        input_scaled = SCALER.transform(input_df)
        
        prediction = MODEL.predict(input_scaled)[0]
        probability = MODEL.predict_proba(input_scaled)[0]
        
        return jsonify({
            'churn': int(prediction),
            'churn_label': LABEL_ENCODER.classes_[prediction],
            'probability': float(probability[1]),
            'confidence': float(max(probability))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Customer Churn Prediction

A machine learning project that predicts customer churn using Logistic Regression with 77% accuracy. Includes a REST API built with Flask for real-time predictions.

## 🎯 Project Overview

- **Dataset**: 7,043 customer records with 21 features
- **Best Model**: Logistic Regression
- **Accuracy**: 77.15%
- **F1-Score**: 0.5729

## 🚀 Features

- Data preprocessing and feature engineering
- Multiple model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- REST API for predictions
- Saved model artifacts for deployment

## 📁 Project Structure
```
customer-churn-prediction/
├── model_output/
│   ├── model_artifacts.pkl       # Trained model
│   ├── model_comparison.csv      # Model performance metrics
│   ├── X_test.npy                # Test features
│   └── y_test.npy                # Test labels
├── preprocess_and_train_simple.py # Training pipeline
├── app.py                         # Flask REST API
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.14
- pip

### Setup

1. Clone the repository
```bash
git clone https://github.com/jenilkathrotia/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Training the Model

Run the training script to preprocess data and train models:
```bash
python preprocess_and_train_simple.py
```

This will:
- Load and preprocess the dataset
- Train 3 different models
- Save the best model to `model_output/`
- Display performance metrics

### Training Output:
```
======================================================================
CUSTOMER CHURN PREDICTION
======================================================================

[1/5] Loading Data...
   Loaded: 7,043 rows, 21 columns

[2/5] Finding Target Column...
   Target: 'Churn'

[3/5] Preprocessing...
   Features: (7043, 6559)

[4/5] Splitting...

[5/5] Training Models...
   Logistic Regression...
      Accuracy: 0.7715, F1: 0.5729
```

## 🌐 API Usage

### Start the API Server
```bash
python app.py
```

The API will run at `http://127.0.0.1:5000/`

### API Endpoints

#### 1. Health Check
```bash
curl http://127.0.0.1:5000/health
```

Response:
```json
{"status": "healthy"}
```

#### 2. Model Information
```bash
curl http://127.0.0.1:5000/
```

Response:
```json
{
  "status": "running",
  "model": "Logistic Regression",
  "accuracy": 0.7715,
  "features": 6559
}
```

#### 3. Predict Churn
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{
  "churn": 0,
  "churn_label": "No",
  "probability": 0.0134,
  "confidence": 0.9866
}
```

## 📈 Model Performance

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | 0.7715   | 0.5729   |
| Random Forest       | 0.7956   | 0.5623   |
| Gradient Boosting   | 0.7935   | 0.5544   |

### Classification Report (Best Model)
```
              precision    recall  f1-score   support
          No       0.85      0.84      0.84      1035
         Yes       0.57      0.58      0.57       374
    accuracy                           0.77      1409
```

## 🔧 Technologies Used

- **Python 3.14**
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **Flask** - REST API framework
- **pickle** - Model serialization

## 📝 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Add feature importance visualization
- [ ] Implement SMOTE for better class balance
- [ ] Create web UI for predictions
- [ ] Deploy to cloud (Heroku/AWS/Render)
- [ ] Add model monitoring and retraining pipeline

## 👤 Author

**Jenil Kathrotia**
- GitHub: [@jenilkathrotia](https://github.com/jenilkathrotia)
- LinkedIn: [Jenil Kathrotia](https://linkedin.com/in/jenilkathrotia)

## 📄 License

This project is open source and available under the MIT License.

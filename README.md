# 🛡️ AI-Powered Cybersecurity Threat Detection System

This project is an end-to-end machine learning system that detects network intrusions using the **NSL-KDD** dataset. It uses a **Random Forest** classifier optimized through cross-validation and hyperparameter tuning, and provides a **Flask API** to serve real-time predictions.

---

## 📊 Dataset Information

- **Dataset**: [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- **Type**: Benchmark dataset for intrusion detection
- **Features**: 41 network connection features
- **Labels**: Normal, and various attack types (DoS, Probe, R2L, U2R)

---

## ✅ Key Features

- Preprocessing & Feature Engineering
- Model training with Cross-Validation
- Hyperparameter tuning via `GridSearchCV`
- Deployment using Flask REST API
- Model artifacts stored using `joblib`
- Optional Web UI using Streamlit
- Cloud deployment support (Render, Heroku, AWS)

---

## 📁 Project Structure

AI-Powered-Cybersecurity
│ ├── nsl_kdd_model.pkl # Trained model
│ ├── scaler.pkl # Scaler used for input normalization
│ └── feature_names.pkl # Ordered list of feature names
├── app.py # Flask API for predictions
├── requirements.txt # Project dependencies
├── test_request.py # Script to test the API endpoint


---

## ⚙️ Model Training Workflow

### 🔹 Preprocessing Steps

- Encode categorical columns: `protocol_type`, `service`, `flag`
- Drop irrelevant feature: `num_outbound_cmds`
- Scale numerical values using `StandardScaler`

### 🔹 Cross-Validation

python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("CV Accuracy:", cv_scores.mean())

🔹 Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_scaled, y)
🔹 Save the Model and Artifacts
python

import joblib
joblib.dump(grid.best_estimator_, "model/nsl_kdd_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

🚀 Running the Flask API
🔹 Start the Server

python app.py
🔹 Endpoint
POST /predict

🔹 Sample Request
json
{
  "duration": 0,
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF",
  "src_bytes": 181,
  "dst_bytes": 5450,
  ...
}

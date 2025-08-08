from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

# Load model and feature list
model = joblib.load("nsl_kdd_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # <-- saved list of 40 features

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        input_features = [data[feature] for feature in feature_names]
        input_array = np.array([input_features])

        # Make prediction
        prediction = model.predict(input_array)
        return jsonify({"Threat_Detected": bool(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)   # 允许跨域请求

# 加载模型和特征列名
model = joblib.load('heart_model.pkl')
feature_names = joblib.load('features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_values = [float(data.get(f, 0)) for f in feature_names]
    input_df = pd.DataFrame([input_values], columns=feature_names)
    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= 0.5)
    return jsonify({'probability': prob, 'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
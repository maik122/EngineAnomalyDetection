# app.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ── Load model and scaler ───────────────────────────────────────────────────
model = tf.keras.models.load_model('gru_model.keras')
scaler = joblib.load('scaler.pkl')
INFERENCE_COLUMNS = joblib.load('inference_columns.pkl')

SEQUENCE_LENGTH = 30

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the uploaded CSV file
    file = request.files.get('csv_file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # 2. Read it into a dataframe
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not read CSV: {e}'}), 400

    # 3. Validate shape
    if df.shape[0] < SEQUENCE_LENGTH:
        return jsonify({
            'error': f'Need at least {SEQUENCE_LENGTH} rows, got {df.shape[0]}'
        }), 400

    missing = [col for col in INFERENCE_COLUMNS if col not in df.columns]
    if missing:
        return jsonify({'error': f'Missing columns: {missing}'}), 400

    # 4. Take the last 30 rows, in the right column order
    df = df[INFERENCE_COLUMNS].tail(SEQUENCE_LENGTH)

    # 5. Scale using the fixed scaler
    scaled = scaler.transform(df)

    # 6. Reshape to (1, 30, 18) — what the GRU expects
    X = scaled.reshape(1, SEQUENCE_LENGTH, len(INFERENCE_COLUMNS))

    # 7. Predict
    prediction = model.predict(X)
    rul = float(prediction[0][0])

    # 8. Return result
    return jsonify({'predicted_rul': round(rul, 1)})


if __name__ == '__main__':
    app.run(debug=True)
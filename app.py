import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('gru_model.keras')

# Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the input features (sensor columns)
sensor_columns = ['Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_7', 'Sensor_8', 'Sensor_9',
                  'Sensor_11', 'Sensor_12', 'Sensor_13', 'Sensor_14', 'Sensor_15', 
                  'Sensor_17', 'Sensor_20', 'Sensor_21']

# Home route
@app.route('/')
def home():
    return render_template('index.html', sensor_columns=sensor_columns)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    sensor_data = []
    for column in sensor_columns:
        sensor_value = float(request.form[column])  # get the value entered in the form
        sensor_data.append(sensor_value)
    
    # Convert the input data into a numpy array
    sensor_data = np.array(sensor_data).reshape(1, -1)  # Reshape for the model

    # Scale the input data using the same scaler from training
    scaled_data = scaler.transform(sensor_data)

    # Make a prediction using the model
    prediction = model.predict(scaled_data)

    # Show the predicted RUL (Remaining Useful Life)
    return render_template('index.html', prediction_text=f'Predicted RUL: {prediction[0][0]:.2f}', sensor_columns=sensor_columns)

if __name__ == '__main__':
    app.run(debug=True)

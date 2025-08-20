# Engine Anomaly Detection Using GRU Model

This project focuses on detecting anomalies in engine data using a **Gated Recurrent Unit (GRU)** neural network. The dataset includes sensor readings and operational settings of engines, and the goal is to predict the **Remaining Useful Life (RUL)** of each engine.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling](#modeling)
4. [Evaluation](#evaluation)
5. [How to Use](#how-to-use)
6. [License](#license)

---

## Project Overview

The project uses sensor data from engines to predict their **Remaining Useful Life (RUL)**. A GRU model was chosen for this task due to its ability to handle time series data effectively. The dataset consists of operational settings and sensor readings over time, with a target variable (`RUL`) indicating how much longer the engine is expected to function.

### Key Tasks:
- **Data Preprocessing:** Cleaning and scaling of the data.
- **Modeling:** Training a GRU model to predict the Remaining Useful Life.
- **Evaluation:** Assessing model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
  
---

## Data Preprocessing

The dataset is preprocessed before training the model. The steps include:

1. **Handling Missing Data:** Any missing values in the dataset were handled appropriately.
2. **Scaling Features:** A `MinMaxScaler` was used to scale the sensor and operational setting columns for better model performance.
3. **Feature Selection:** Only relevant columns (sensor readings and operational settings) were selected for training the model.
4. **Train-Test Split:** The dataset was split into a training set and a testing set to evaluate the model's performance.

### Important Columns:
- **`RUL`:** The remaining useful life of the engine, which is the target variable.
- **Sensor Columns:** Various sensors (`Sensor_2`, `Sensor_3`, etc.) provide engine state data.
- **Operational Settings:** Parameters like `Op_Setting_1`, `Op_Setting_2`, and `Op_Setting_3` are also part of the data used to make predictions.

---

## Modeling

The **GRU model** is used to predict the Remaining Useful Life of the engines. Here's a brief overview of the steps followed to train the model:

1. **Model Definition:** A simple GRU model was created with the following layers:
   - **Input layer** to accept time series data.
   - **GRU layers** for capturing temporal patterns in the data.
   - **Dense layer** to output the predicted RUL.
   
2. **Compilation:** The model was compiled using the **Mean Squared Error (MSE)** loss function and **Adam optimizer**.

3. **Training:** The model was trained on the preprocessed training data.

4. **Evaluation:** After training, the model was evaluated on the validation set using several metrics:
   - **Mean Absolute Error (MAE):** 17.84
   - **Root Mean Squared Error (RMSE):** 24.09
   - **R² Score:** 0.83

---

## Evaluation

The GRU model was evaluated using three common regression evaluation metrics:

- **Mean Absolute Error (MAE):** This metric measures the average of the absolute errors between the predicted and actual RUL values. A lower MAE indicates better model performance.
- **Root Mean Squared Error (RMSE):** This metric gives more weight to large errors, penalizing the model more if it makes larger mistakes.
- **R² Score:** This indicates how well the model explains the variance in the target variable. A higher R² score is desirable.

The evaluation metrics on the validation set are as follows:
- MAE: 17.84
- RMSE: 24.09
- R² Score: 0.83

These results suggest that the model performs quite well, with a reasonable prediction of the remaining useful life.

---


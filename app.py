# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD FILES
# =========================

model = joblib.load("svr_model.pkl")

scaler_X = joblib.load("scaler_X.pkl")

scaler_y = joblib.load("scaler_y.pkl")

label_encoders = joblib.load("label_encoders.pkl")

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("CarPrice_Assignment.csv")

# Remove unnecessary columns
columns_to_remove = ['car_ID', 'CarName']

for col in columns_to_remove:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =========================
# STREAMLIT PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🚗 Car Price Prediction using SVR")

st.write("Enter Car Details Below")

# =========================
# FEATURES
# =========================

target_column = 'price'

feature_columns = df.drop(target_column, axis=1).columns

input_data = []

# =========================
# USER INPUTS
# =========================

for column in feature_columns:

    # Categorical Columns
    if column in label_encoders:

        options = df[column].unique()

        selected_value = st.selectbox(
            f"Select {column}",
            options
        )

        encoder = label_encoders[column]

        encoded_value = encoder.transform([selected_value])[0]

        input_data.append(encoded_value)

    # Numerical Columns
    else:

        value = st.number_input(
            f"Enter {column}",
            value=float(df[column].mean())
        )

        input_data.append(value)

# =========================
# PREDICTION
# =========================

if st.button("Predict Car Price"):

    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler_X.transform(input_array)

    # Predict scaled value
    prediction_scaled = model.predict(input_scaled)

    # Convert back to original price
    prediction = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1,1)
    )

    st.success(
        f"Estimated Car Price: ${prediction[0][0]:,.2f}"
    )
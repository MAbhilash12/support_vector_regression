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
# UI LAYOUT
# =========================

col1, col2 = st.columns(2)

# =========================
# COLUMN 1
# =========================

with col1:

    fueltype = st.selectbox(
        "Fuel Type",
        df['fueltype'].unique()
    )

    carbody = st.selectbox(
        "Car Body",
        df['carbody'].unique()
    )

    horsepower = st.slider(
        "Horsepower",
        int(df['horsepower'].min()),
        int(df['horsepower'].max()),
        int(df['horsepower'].mean())
    )

    enginesize = st.slider(
        "Engine Size",
        int(df['enginesize'].min()),
        int(df['enginesize'].max()),
        int(df['enginesize'].mean())
    )

# =========================
# COLUMN 2
# =========================

with col2:

    drivewheel = st.selectbox(
        "Drive Wheel",
        df['drivewheel'].unique()
    )

    curbweight = st.slider(
        "Curb Weight",
        int(df['curbweight'].min()),
        int(df['curbweight'].max()),
        int(df['curbweight'].mean())
    )

    citympg = st.slider(
        "City MPG",
        int(df['citympg'].min()),
        int(df['citympg'].max()),
        int(df['citympg'].mean())
    )

    highwaympg = st.slider(
        "Highway MPG",
        int(df['highwaympg'].min()),
        int(df['highwaympg'].max()),
        int(df['highwaympg'].mean())
    )

# =========================
# ENCODE CATEGORICAL VALUES
# =========================

fueltype_encoded = label_encoders['fueltype'].transform([fueltype])[0]

carbody_encoded = label_encoders['carbody'].transform([carbody])[0]

drivewheel_encoded = label_encoders['drivewheel'].transform([drivewheel])[0]

# =========================
# CREATE INPUT ARRAY
# =========================

input_data = np.array([
    [
        fueltype_encoded,
        carbody_encoded,
        drivewheel_encoded,
        horsepower,
        enginesize,
        curbweight,
        citympg,
        highwaympg
    ]
])

# =========================
# PREDICTION
# =========================

if st.button("Predict Car Price"):

    # Scale input
    input_scaled = scaler_X.transform(input_data)

    # Predict scaled value
    prediction_scaled = model.predict(input_scaled)

    # Convert back to original price
    prediction = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1,1)
    )

    st.success(
        f"Estimated Car Price: ${prediction[0][0]:,.2f}"
    )

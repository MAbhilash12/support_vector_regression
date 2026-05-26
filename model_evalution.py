# model_evaluation.py

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD FILES
# =========================

model = joblib.load("svr_model.pkl")

scaler_X = joblib.load("scaler_X.pkl")

scaler_y = joblib.load("scaler_y.pkl")

X_test = joblib.load("X_test.pkl")

y_test = joblib.load("y_test.pkl")

# =========================
# SCALE TEST DATA
# =========================

X_test_scaled = scaler_X.transform(X_test)

# =========================
# PREDICTIONS
# =========================

y_pred_scaled = model.predict(X_test_scaled)

# Convert back to original values
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1,1)
).ravel()

# =========================
# METRICS
# =========================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========\n")

print(f"MAE       : {mae:.2f}")

print(f"MSE       : {mse:.2f}")

print(f"RMSE      : {rmse:.2f}")

print(f"R2 Score  : {r2:.2f}")

# =========================
# ACTUAL VS PREDICTED GRAPH
# =========================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()

# =========================
# RESIDUAL PLOT
# =========================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Prices")

plt.ylabel("Residual Errors")

plt.title("Residual Plot")

plt.show()

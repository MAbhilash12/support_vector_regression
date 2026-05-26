# data_preprocessing.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("CarPrice_Assignment.csv")

print("Dataset Loaded Successfully!")

# =========================
# REMOVE UNNECESSARY COLUMNS
# =========================

columns_to_remove = ['car_ID', 'CarName']

for col in columns_to_remove:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =========================
# HANDLE MISSING VALUES
# =========================

df.dropna(inplace=True)

# =========================
# LABEL ENCODING
# =========================

label_encoders = {}

categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    label_encoders[column] = le

# Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")

print("Categorical Columns Encoded!")

# =========================
# TARGET COLUMN
# =========================

target_column = 'price'

# =========================
# IMPORTANT FEATURES ONLY
# =========================

selected_features = [
    'fueltype',
    'carbody',
    'drivewheel',
    'horsepower',
    'enginesize',
    'curbweight',
    'citympg',
    'highwaympg'
]

X = df[selected_features]

y = df[target_column]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save test data
joblib.dump(X_test, "X_test.pkl")

joblib.dump(y_test, "y_test.pkl")

# =========================
# FEATURE SCALING
# =========================

scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)

X_test_scaled = scaler_X.transform(X_test)

# Scale target values
scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(
    y_train.values.reshape(-1,1)
).ravel()

# Save scalers
joblib.dump(scaler_X, "scaler_X.pkl")

joblib.dump(scaler_y, "scaler_y.pkl")

print("Feature Scaling Completed!")

# =========================
# TRAIN SVR MODEL
# =========================

model = SVR(kernel='rbf')

model.fit(X_train_scaled, y_train_scaled)

# Save model
joblib.dump(model, "svr_model.pkl")

print("SVR Model Trained Successfully!")

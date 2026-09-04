import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. FILE PATHS
# =========================================================

DATASET_PATH = "dataset/house_price_dataset.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")


# =========================================================
# 2. CREATE MODEL FOLDER
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# =========================================================
# 4. DISPLAY DATASET INFORMATION
# =========================================================

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# =========================================================
# 5. REMOVE UNNECESSARY INDEX COLUMN IF PRESENT
# =========================================================

if "index" in df.columns:
    df = df.drop(columns=["index"])


# =========================================================
# 6. DEFINE FEATURES AND TARGET
# =========================================================

features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking",
    "Age",
    "Location"
]

target = "Price"

X = df[features]
y = df[target]


# =========================================================
# 7. DEFINE COLUMN TYPES
# =========================================================

numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking",
    "Age"
]

categorical_features = [
    "Location"
]


# =========================================================
# 8. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 9. CREATE RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 10. CREATE COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# 11. SPLIT DATA INTO TRAINING AND TESTING
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 12. TRAIN THE MODEL
# =========================================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# =========================================================
# 13. MAKE PREDICTIONS
# =========================================================

y_pred = pipeline.predict(X_test)


# =========================================================
# 14. CALCULATE MODEL PERFORMANCE
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


# =========================================================
# 15. DISPLAY RESULTS
# =========================================================

print("\n========================================")
print("       MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : ₹{mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R² Score : {r2:.4f}")

print("========================================")


# =========================================================
# 16. SAVE TRAINED MODEL
# =========================================================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)

print("\nTraining process completed successfully! 🎉")
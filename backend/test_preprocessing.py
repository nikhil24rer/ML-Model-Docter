
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.preprocessing import (
    build_preprocessor,
    get_preprocessing_summary
)


# ==========================================
# TEST DATASET
# ==========================================

X = pd.DataFrame({

    "Age": [21, 22, 24, 25, 65, 30, np.nan, 35],

    "Salary": [
        25000,
        30000,
        35000,
        40000,
        500000,
        45000,
        55000,
        np.inf
    ],

    "City": [
        "Ludhiana",
        "Delhi",
        "Mohali",
        "Chandigarh",
        "Delhi",
        "Ludhiana",
        None,
        "Mohali"
    ],

    "Active": [
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True
    ]

})


# ==========================================
# TEST 1: RANDOM FOREST PREPROCESSING
# ==========================================

print("\nTEST 1: Random Forest Preprocessing")

preprocessor = build_preprocessor(
    X,
    algorithm="random_forest"
)

X_transformed = preprocessor.fit_transform(X)

print("Original Shape:", X.shape)

print("Transformed Shape:", X_transformed.shape)

print("Scaling:", False)


# ==========================================
# TEST 2: LOGISTIC REGRESSION PREPROCESSING
# ==========================================

print("\nTEST 2: Logistic Regression Preprocessing")

preprocessor2 = build_preprocessor(
    X,
    algorithm="logistic_regression"
)

X_transformed2 = preprocessor2.fit_transform(X)

print("Original Shape:", X.shape)

print("Transformed Shape:", X_transformed2.shape)

print("Scaling:", True)


# ==========================================
# TEST 3: CHECK MISSING VALUE HANDLING
# ==========================================

print("\nTEST 3: Missing Value Handling")

assert np.isfinite(
    X_transformed.toarray()
    if hasattr(X_transformed, "toarray")
    else X_transformed
).all()

print("No missing or infinite values after transformation.")


# ==========================================
# TEST 4: UNKNOWN CATEGORY HANDLING
# ==========================================

print("\nTEST 4: Unknown Category Handling")

new_data = pd.DataFrame({

    "Age": [28],

    "Salary": [60000],

    "City": ["New City"],

    "Active": [True]

})

new_transformed = preprocessor.transform(new_data)

print("New category processed successfully.")

print("Transformed Shape:", new_transformed.shape)


# ==========================================
# TEST 5: PREPROCESSING SUMMARY
# ==========================================

print("\nTEST 5: Preprocessing Summary")

summary = get_preprocessing_summary(
    X,
    algorithm="random_forest"
)

for key, value in summary.items():

    print(key, ":", value)


# ==========================================
# TEST 6: COMPLETE ML PIPELINE
# ==========================================

print("\nTEST 6: Complete ML Pipeline")

y = pd.Series([
    0, 0, 1, 1, 1, 0, 0, 1
])

model_pipeline = Pipeline([

    (
        "preprocessor",
        build_preprocessor(
            X,
            algorithm="random_forest"
        )
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
    )

])

model_pipeline.fit(X, y)

predictions = model_pipeline.predict(new_data)

print("Prediction:", predictions)

print("Complete pipeline trained successfully.")


# ==========================================
# FINAL RESULT
# ==========================================

print("\n========== ALL TESTS COMPLETED ==========")

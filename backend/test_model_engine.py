
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression

from ml.model_engine import (
    detect_problem_type,
    build_model_pipeline,
    evaluate_classification,
    evaluate_regression,
    compare_models,
)


# ==================================================
# TEST 1: TARGET DETECTION
# ==================================================

print("\nTEST 1: Target Detection")

target = pd.Series(
    ["Yes", "No", "Yes", "No", "Yes", "No"] * 5,
    name="Purchased",
)

problem_type = detect_problem_type(target)

print("Detected problem type:", problem_type)

assert problem_type == "binary_classification"


# ==================================================
# TEST 2: CLASSIFICATION MODEL PIPELINE
# ==================================================

print("\nTEST 2: Classification Pipeline")

X_class, y_class = make_classification(
    n_samples=120,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

X_class = pd.DataFrame(
    X_class,
    columns=["Feature_1", "Feature_2", "Feature_3", "Feature_4"],
)

X_class["City"] = np.random.default_rng(42).choice(
    ["Delhi", "Mumbai", "Ludhiana"],
    size=120,
)

X_class["Active"] = np.random.default_rng(42).choice(
    [True, False],
    size=120,
)

y_class = pd.Series(y_class, name="Target")

pipeline = build_model_pipeline(
    X_class.iloc[:96],
    "random_forest",
    "classification",
)

pipeline.fit(
    X_class.iloc[:96],
    y_class.iloc[:96],
)

predictions = pipeline.predict(X_class.iloc[96:])

print("Predictions:", predictions[:10])
print("Pipeline trained successfully.")

assert len(predictions) == 24


# ==================================================
# TEST 3: CLASSIFICATION METRICS
# ==================================================

print("\nTEST 3: Classification Metrics")

metrics = evaluate_classification(
    y_class.iloc[96:],
    predictions,
)

print(metrics)

assert "accuracy" in metrics
assert "f1" in metrics
assert "confusion_matrix" in metrics


# ==================================================
# TEST 4: COMPLETE CLASSIFICATION COMPARISON
# ==================================================

print("\nTEST 4: Complete Classification Comparison")

classification_result = compare_models(
    X_class,
    y_class,
    problem_type="classification",
    max_models=6,
    test_size=0.20,
)

print("Problem type:", classification_result["problem_type"])
print("Best model:", classification_result["best_model"])
print("Best score:", classification_result["best_score"])
print("Baseline:", classification_result["baseline"])
print("Successful models:", classification_result["successful_models"])
print("Failed models:", classification_result["failed_models"])

assert classification_result["successful_models"] >= 1
assert classification_result["best_model_id"] in (
    classification_result["_trained_models"]
)

assert classification_result["baseline"] is not None


# ==================================================
# TEST 5: REGRESSION MODEL PIPELINE
# ==================================================

print("\nTEST 5: Regression Pipeline")

X_reg, y_reg = make_regression(
    n_samples=120,
    n_features=4,
    noise=10,
    random_state=42,
)

X_reg = pd.DataFrame(
    X_reg,
    columns=["Feature_1", "Feature_2", "Feature_3", "Feature_4"],
)

X_reg["Category"] = np.random.default_rng(21).choice(
    ["A", "B", "C"],
    size=120,
)

y_reg = pd.Series(y_reg, name="Target")

pipeline = build_model_pipeline(
    X_reg.iloc[:96],
    "random_forest_regressor",
    "regression",
)

pipeline.fit(
    X_reg.iloc[:96],
    y_reg.iloc[:96],
)

predictions = pipeline.predict(X_reg.iloc[96:])

print("Predictions:", predictions[:10])
print("Regression pipeline trained successfully.")

assert len(predictions) == 24


# ==================================================
# TEST 6: REGRESSION METRICS
# ==================================================

print("\nTEST 6: Regression Metrics")

metrics = evaluate_regression(
    y_reg.iloc[96:],
    predictions,
)

print(metrics)

assert "r2" in metrics
assert "mae" in metrics
assert "rmse" in metrics


# ==================================================
# TEST 7: COMPLETE REGRESSION COMPARISON
# ==================================================

print("\nTEST 7: Complete Regression Comparison")

regression_result = compare_models(
    X_reg,
    y_reg,
    problem_type="regression",
    max_models=6,
    test_size=0.20,
)

print("Problem type:", regression_result["problem_type"])
print("Best model:", regression_result["best_model"])
print("Best score:", regression_result["best_score"])
print("Baseline:", regression_result["baseline"])
print("Successful models:", regression_result["successful_models"])
print("Failed models:", regression_result["failed_models"])

assert regression_result["successful_models"] >= 1
assert regression_result["best_model_id"] in (
    regression_result["_trained_models"]
)


# ==================================================
# TEST 8: INVALID INPUT HANDLING
# ==================================================

print("\nTEST 8: Invalid Input Handling")

try:
    compare_models(
        pd.DataFrame({"A": [1, 2, 3]}),
        pd.Series([1, 2]),
        problem_type="regression",
    )

    raise AssertionError("Expected ValueError was not raised.")

except ValueError as error:
    print("Correctly handled:", error)


# ==================================================
# TEST 9: EMPTY FEATURE COLUMN HANDLING
# ==================================================

print("\nTEST 9: Empty Feature Column Handling")

X_with_empty = X_reg.copy()
X_with_empty["Empty_Column"] = np.nan

empty_result = compare_models(
    X_with_empty,
    y_reg,
    problem_type="regression",
    max_models=2,
)

print(
    "Removed columns:",
    empty_result["dataset_info"]["removed_empty_columns"],
)

assert "Empty_Column" in (
    empty_result["dataset_info"]["removed_empty_columns"]
)


print("\n========== ALL TESTS COMPLETED ==========")

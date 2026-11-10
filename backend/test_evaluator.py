
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from ml.evaluator import (
    evaluate_classification_model,
    evaluate_regression_model,
    compare_train_test_performance,
    compare_with_baseline,
    detect_underfitting,
    build_evaluation_report,
    validate_predictions,
)


# ==================================================
# TEST 1: CLASSIFICATION METRICS
# ==================================================

print("\nTEST 1: Classification Metrics")

actual = [0, 1, 0, 1, 0, 1, 1, 0]
predicted = [0, 1, 0, 0, 0, 1, 1, 1]

classification_metrics = evaluate_classification_model(
    actual,
    predicted,
)

print(classification_metrics)

assert "accuracy" in classification_metrics
assert "balanced_accuracy" in classification_metrics
assert "precision" in classification_metrics
assert "recall" in classification_metrics
assert "f1" in classification_metrics
assert "confusion_matrix" in classification_metrics
assert "class_wise_metrics" in classification_metrics

print("Classification metrics passed.")


# ==================================================
# TEST 2: REGRESSION METRICS
# ==================================================

print("\nTEST 2: Regression Metrics")

actual = [100, 200, 300, 400, 500]
predicted = [110, 190, 310, 390, 510]

regression_metrics = evaluate_regression_model(
    actual,
    predicted,
)

print(regression_metrics)

assert "r2" in regression_metrics
assert "mae" in regression_metrics
assert "rmse" in regression_metrics
assert "median_absolute_error" in regression_metrics
assert "mean_residual" in regression_metrics

print("Regression metrics passed.")


# ==================================================
# TEST 3: TRAIN-TEST PERFORMANCE COMPARISON
# ==================================================

print("\nTEST 3: Generalization Gap")

train_metrics = {
    "f1": 0.98,
}

test_metrics = {
    "f1": 0.75,
}

generalization = compare_train_test_performance(
    train_metrics,
    test_metrics,
    "classification",
)

print(generalization)

assert generalization["overfitting_suspected"] is True
assert generalization["performance_gap"] == 0.23

print("Generalization check passed.")


# ==================================================
# TEST 4: BASELINE COMPARISON
# ==================================================

print("\nTEST 4: Baseline Comparison")

model_metrics = {
    "accuracy": 0.90,
    "balanced_accuracy": 0.89,
    "f1": 0.88,
}

baseline_metrics = {
    "accuracy": 0.60,
    "balanced_accuracy": 0.50,
    "f1": 0.45,
}

baseline_comparison = compare_with_baseline(
    model_metrics,
    baseline_metrics,
    "classification",
)

print(baseline_comparison)

assert baseline_comparison["status"] == "better_than_baseline"

print("Baseline comparison passed.")


# ==================================================
# TEST 5: UNDERFITTING INDICATOR
# ==================================================

print("\nTEST 5: Underfitting Indicator")

train_metrics = {
    "f1": 0.40,
}

baseline_train_metrics = {
    "f1": 0.50,
}

underfitting = detect_underfitting(
    train_metrics,
    baseline_train_metrics,
    "classification",
)

print(underfitting)

assert underfitting["underfitting_suspected"] is True

print("Underfitting check passed.")


# ==================================================
# TEST 6: COMPLETE CLASSIFICATION REPORT
# ==================================================

print("\nTEST 6: Complete Classification Report")

X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_train_predictions = baseline.predict(X_train)
baseline_test_predictions = baseline.predict(X_test)

train_metrics = evaluate_classification_model(
    y_train,
    train_predictions,
)

test_metrics = evaluate_classification_model(
    y_test,
    test_predictions,
)

baseline_train_metrics = evaluate_classification_model(
    y_train,
    baseline_train_predictions,
)

baseline_test_metrics = evaluate_classification_model(
    y_test,
    baseline_test_predictions,
)

report = build_evaluation_report(
    train_metrics,
    test_metrics,
    baseline_train_metrics,
    baseline_test_metrics,
    "classification",
)

print("Generalization:", report["generalization"])
print("Baseline:", report["baseline_comparison"])
print("Underfitting:", report["underfitting"])
print("Warnings:", report["warnings"])

assert report["problem_type"] == "classification"
assert "warnings" in report

print("Complete classification report passed.")


# ==================================================
# TEST 7: COMPLETE REGRESSION REPORT
# ==================================================

print("\nTEST 7: Complete Regression Report")

X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_train_predictions = baseline.predict(X_train)
baseline_test_predictions = baseline.predict(X_test)

train_metrics = evaluate_regression_model(
    y_train,
    train_predictions,
)

test_metrics = evaluate_regression_model(
    y_test,
    test_predictions,
)

baseline_train_metrics = evaluate_regression_model(
    y_train,
    baseline_train_predictions,
)

baseline_test_metrics = evaluate_regression_model(
    y_test,
    baseline_test_predictions,
)

report = build_evaluation_report(
    train_metrics,
    test_metrics,
    baseline_train_metrics,
    baseline_test_metrics,
    "regression",
)

print("Generalization:", report["generalization"])
print("Baseline:", report["baseline_comparison"])
print("Underfitting:", report["underfitting"])
print("Warnings:", report["warnings"])

assert report["problem_type"] == "regression"
assert "r2" in report["test_metrics"]

print("Complete regression report passed.")


# ==================================================
# TEST 8: INVALID INPUT HANDLING
# ==================================================

print("\nTEST 8: Invalid Input Handling")

try:
    validate_predictions(
        [1, 2, 3],
        [1, 2],
    )

    raise AssertionError("Expected ValueError was not raised.")

except ValueError as error:
    print("Correctly handled:", error)


print("\n========== ALL TESTS COMPLETED ==========")

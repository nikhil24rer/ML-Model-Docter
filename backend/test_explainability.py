
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.explainability import (
    build_importance_summary,
    get_native_feature_importance,
    get_model_coefficients,
    get_permutation_importance,
    explain_prediction,
    explain_model,
)


print("\nTEST 1: Feature Importance Summary")

summary = build_importance_summary(
    [0.2, 0.6, 0.2],
    ["Age", "Salary", "Experience"],
)

assert summary["top_features"][0]["feature"] == "Salary"
assert summary["total_features"] == 3

print("Feature importance summary passed.")


print("\nTEST 2: Native Feature Importance")

X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    random_state=42,
)

X = pd.DataFrame(
    X,
    columns=["A", "B", "C", "D", "E"],
)

model = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
)

model.fit(X, y)

result = get_native_feature_importance(
    model,
    feature_names=X.columns,
)

assert result["available"] is True
assert result["total_features"] == 5

print("Native importance passed.")


print("\nTEST 3: Logistic Regression Coefficients")

linear_model = LogisticRegression(max_iter=1000)
linear_model.fit(X, y)

result = get_model_coefficients(
    linear_model,
    feature_names=X.columns,
)

assert result["available"] is True
assert result["total_features"] == 5

print("Coefficient extraction passed.")


print("\nTEST 4: Permutation Importance")

result = get_permutation_importance(
    model,
    X,
    y,
    n_repeats=3,
)

assert result["available"] is True
assert result["total_features"] == 5

print("Permutation importance passed.")


print("\nTEST 5: Individual Prediction")

result = explain_prediction(
    model,
    X,
    row_index=0,
)

assert "prediction" in result
assert "input_features" in result

print("Prediction:", result["prediction"])
print("Individual prediction explanation passed.")


print("\nTEST 6: Complete Explainability Report")

report = explain_model(
    model,
    feature_names=X.columns,
)

assert report["available"] is True
assert "native_feature_importance" in report["methods"]

print("Methods:", report["methods"])
print("Complete report passed.")


print("\nTEST 7: Unsupported Native Importance")

result = get_native_feature_importance(linear_model)

assert result["available"] is False

print("Unsupported model handled correctly.")


print("\nTEST 8: Pipeline Compatibility")

pipeline = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    ))
])

pipeline.fit(X, y)

result = get_native_feature_importance(
    pipeline,
    feature_names=X.columns,
)

assert result["available"] is True

print("Pipeline compatibility passed.")


print("\nTEST 9: Invalid Input Handling")

try:
    build_importance_summary([0.2, 0.3], ["Age"])
    raise AssertionError("Expected ValueError")

except ValueError:
    print("Mismatched feature names handled correctly.")


print("\nTEST 10: Empty Input Handling")

try:
    get_permutation_importance(model, np.empty((0, 5)), [])
    raise AssertionError("Expected ValueError")

except ValueError:
    print("Empty input handled correctly.")


print("\n========== ALL EXPLAINABILITY TESTS COMPLETED ==========")

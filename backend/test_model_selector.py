
from ml.model_selector import (
    select_models,
    select_classification_models,
    select_regression_models,
)


print("\nTEST 1: Small Classification Dataset")

result = select_models(
    n_samples=80,
    n_features=5,
    problem_type="classification",
    n_classes=2,
)

print("Selected models:", result["selected_models"])
print("Reasons:", result["selection_reasons"])

assert "logistic_regression" in result["selected_models"]
assert result["problem_type"] == "classification"

print("Small classification test passed.")


print("\nTEST 2: Large Classification Dataset")

result = select_models(
    n_samples=25000,
    n_features=15,
    problem_type="classification",
    n_classes=3,
)

print("Selected models:", result["selected_models"])

assert "svm" not in result["selected_models"]
assert "random_forest" in result["selected_models"]

print("Large classification test passed.")


print("\nTEST 3: Imbalanced Classification Dataset")

result = select_models(
    n_samples=1000,
    n_features=10,
    problem_type="classification",
    n_classes=2,
    is_imbalanced=True,
)

print("Selected models:", result["selected_models"])
print("Reasons:", result["selection_reasons"])

assert any(
    "imbalance" in reason.lower()
    for reason in result["selection_reasons"]
)

print("Imbalanced classification test passed.")


print("\nTEST 4: Regression Dataset")

result = select_models(
    n_samples=500,
    n_features=8,
    problem_type="regression",
)

print("Selected models:", result["selected_models"])

assert "linear_regression" in result["selected_models"]
assert "random_forest_regressor" in result["selected_models"]

print("Regression test passed.")


print("\nTEST 5: High-Dimensional Classification")

result = select_models(
    n_samples=100,
    n_features=1500,
    problem_type="classification",
    n_classes=2,
)

print("Selected models:", result["selected_models"])
print("Reasons:", result["selection_reasons"])

assert "logistic_regression" in result["selected_models"]
assert "gradient_boosting" not in result["selected_models"]

print("High-dimensional classification test passed.")


print("\nTEST 6: Maximum Model Limit")

result = select_models(
    n_samples=1000,
    n_features=10,
    problem_type="classification",
    max_models=3,
)

print("Selected models:", result["selected_models"])

assert len(result["selected_models"]) <= 3

print("Maximum model limit test passed.")


print("\nTEST 7: Model Metadata")

result = select_models(
    n_samples=500,
    n_features=10,
    problem_type="classification",
)

details = result["model_details"]

assert len(details) > 0

for model_name, info in details.items():
    print(model_name, ":", info["name"])

print("Model metadata test passed.")


print("\nTEST 8: Invalid Input Handling")

try:
    select_models(
        n_samples=0,
        n_features=5,
        problem_type="classification",
    )
    raise AssertionError("Invalid sample count was accepted.")

except ValueError:
    print("Invalid sample count correctly rejected.")


try:
    select_models(
        n_samples=100,
        n_features=5,
        problem_type="clustering",
    )
    raise AssertionError("Invalid problem type was accepted.")

except ValueError:
    print("Invalid problem type correctly rejected.")


print("\n========== ALL TESTS PASSED ==========")

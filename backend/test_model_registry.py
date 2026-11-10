
from ml.model_registry import (
    get_model_registry,
    create_model,
    get_model_info,
    get_available_models,
)

from ml.model_selector import select_models


# ==================================================
# TEST 1: CLASSIFICATION REGISTRY
# ==================================================

print("\nTEST 1: Classification Registry")

classification = get_model_registry("classification")

print("Total classification models:", len(classification))

for model_id, details in classification.items():
    print(model_id, "->", details["name"])

assert len(classification) == 8


# ==================================================
# TEST 2: REGRESSION REGISTRY
# ==================================================

print("\nTEST 2: Regression Registry")

regression = get_model_registry("regression")

print("Total regression models:", len(regression))

for model_id, details in regression.items():
    print(model_id, "->", details["name"])

assert len(regression) == 8


# ==================================================
# TEST 3: REQUIRED METADATA
# ==================================================

print("\nTEST 3: Model Metadata Validation")

required_keys = {
    "name",
    "estimator",
    "description",
    "category",
    "requires_scaling",
    "supports_probability",
    "default_params",
}

for registry in (classification, regression):
    for model_id, details in registry.items():
        assert required_keys.issubset(details.keys()), (
            f"Missing metadata in {model_id}"
        )

print("All model metadata is available.")


# ==================================================
# TEST 4: CREATE CLASSIFICATION MODELS
# ==================================================

print("\nTEST 4: Create Classification Models")

for model_id in classification:
    model = create_model(
        model_id,
        "classification",
    )

    print(model_id, "->", type(model).__name__)


# ==================================================
# TEST 5: CREATE REGRESSION MODELS
# ==================================================

print("\nTEST 5: Create Regression Models")

for model_id in regression:
    model = create_model(
        model_id,
        "regression",
    )

    print(model_id, "->", type(model).__name__)


# ==================================================
# TEST 6: CUSTOM PARAMETERS
# ==================================================

print("\nTEST 6: Custom Parameters")

model = create_model(
    "random_forest",
    "classification",
    n_estimators=50,
    max_depth=5,
)

assert model.n_estimators == 50
assert model.max_depth == 5

print("Custom parameters applied successfully.")


# ==================================================
# TEST 7: MODEL INFORMATION
# ==================================================

print("\nTEST 7: Model Information")

info = get_model_info(
    "random_forest",
    "classification",
)

print(info)

assert info["id"] == "random_forest"
assert "requires_scaling" in info
assert "supports_probability" in info


# ==================================================
# TEST 8: AVAILABLE MODELS
# ==================================================

print("\nTEST 8: Available Models")

available = get_available_models("classification")

print("Available:", len(available))

assert len(available) == 8


# ==================================================
# TEST 9: MODEL SELECTOR COMPATIBILITY
# ==================================================

print("\nTEST 9: Model Selector Compatibility")

classification_selection = select_models(
    n_samples=200,
    n_features=5,
    problem_type="classification",
    n_classes=2,
    is_imbalanced=False,
    max_models=6,
)

for model_id in classification_selection["selected_models"]:
    assert model_id in classification

print(
    "Classification selected:",
    classification_selection["selected_models"]
)


regression_selection = select_models(
    n_samples=200,
    n_features=5,
    problem_type="regression",
    max_models=6,
)

for model_id in regression_selection["selected_models"]:
    assert model_id in regression

print(
    "Regression selected:",
    regression_selection["selected_models"]
)

print("Selector and registry are compatible.")


# ==================================================
# TEST 10: INVALID MODEL HANDLING
# ==================================================

print("\nTEST 10: Invalid Model Handling")

try:
    create_model(
        "unknown_model",
        "classification",
    )

    raise AssertionError("Expected ValueError was not raised.")

except ValueError as error:
    print("Correctly handled:", error)


print("\n========== ALL TESTS COMPLETED ==========")

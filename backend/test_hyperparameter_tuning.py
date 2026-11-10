
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.hyperparameter_tuning import (
    build_cv_strategy,
    get_default_scoring,
    normalize_param_grid,
    tune_with_grid_search,
    tune_with_random_search,
    tune_model,
    get_best_model,
)


# ============================================================
# TEST 1: DEFAULT SCORING
# ============================================================

print("\nTEST 1: Default Scoring")

assert get_default_scoring("classification") == "f1_macro"
assert get_default_scoring("regression") == "r2"

print("Default scoring passed.")


# ============================================================
# TEST 2: CV STRATEGY
# ============================================================

print("\nTEST 2: CV Strategy")

X_cls, y_cls = make_classification(
    n_samples=150,
    n_features=5,
    random_state=42,
)

cv_strategy = build_cv_strategy(
    "classification",
    y_cls,
    cv=3,
)

assert cv_strategy.n_splits == 3

print("CV strategy passed.")


# ============================================================
# TEST 3: PIPELINE PARAMETER NORMALIZATION
# ============================================================

print("\nTEST 3: Pipeline Parameter Normalization")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

normalized = normalize_param_grid(
    pipeline,
    {"C": [0.1, 1.0]},
)

assert "model__C" in normalized

print("Pipeline parameter normalization passed.")


# ============================================================
# TEST 4: GRID SEARCH CLASSIFICATION
# ============================================================

print("\nTEST 4: Grid Search Classification")

model = LogisticRegression(max_iter=1000)

report = tune_with_grid_search(
    model=model,
    X=X_cls,
    y=y_cls,
    param_grid={
        "C": [0.1, 1.0, 10.0],
    },
    problem_type="classification",
    cv=3,
    n_jobs=1,
)

assert report["search_method"] == "grid_search"
assert report["best_estimator"] is not None
assert report["total_configurations_tested"] == 3

print("Best parameters:", report["best_params"])
print("Best CV score:", report["best_score"])
print("Grid search classification passed.")


# ============================================================
# TEST 5: RANDOM SEARCH CLASSIFICATION
# ============================================================

print("\nTEST 5: Random Search Classification")

model = RandomForestClassifier(random_state=42)

report = tune_with_random_search(
    model=model,
    X=X_cls,
    y=y_cls,
    param_distributions={
        "n_estimators": [20, 30, 40],
        "max_depth": [None, 3, 5],
    },
    problem_type="classification",
    cv=3,
    n_iter=4,
    n_jobs=1,
)

assert report["search_method"] == "randomized_search"
assert report["best_estimator"] is not None
assert report["total_configurations_tested"] == 4

print("Best parameters:", report["best_params"])
print("Random search classification passed.")


# ============================================================
# TEST 6: REGRESSION GRID SEARCH
# ============================================================

print("\nTEST 6: Regression Grid Search")

X_reg, y_reg = make_regression(
    n_samples=150,
    n_features=5,
    noise=10,
    random_state=42,
)

model = Ridge()

report = tune_with_grid_search(
    model=model,
    X=X_reg,
    y=y_reg,
    param_grid={
        "alpha": [0.1, 1.0, 10.0],
    },
    problem_type="regression",
    cv=3,
    n_jobs=1,
)

assert report["search_method"] == "grid_search"
assert report["best_estimator"] is not None

print("Best parameters:", report["best_params"])
print("Regression grid search passed.")


# ============================================================
# TEST 7: UNIFIED TUNING FUNCTION
# ============================================================

print("\nTEST 7: Unified Tuning")

report = tune_model(
    model=LogisticRegression(max_iter=1000),
    X=X_cls,
    y=y_cls,
    param_grid={"C": [0.1, 1.0]},
    problem_type="classification",
    method="grid",
    cv=3,
    n_jobs=1,
)

assert report["search_method"] == "grid_search"

print("Unified tuning passed.")


# ============================================================
# TEST 8: BEST MODEL EXTRACTION
# ============================================================

print("\nTEST 8: Best Model Extraction")

best_model = get_best_model(report)

assert hasattr(best_model, "predict")

print("Best model extraction passed.")


# ============================================================
# TEST 9: INVALID PARAMETER HANDLING
# ============================================================

print("\nTEST 9: Invalid Parameters")

try:
    tune_with_grid_search(
        model=LogisticRegression(),
        X=X_cls,
        y=y_cls,
        param_grid={},
        problem_type="classification",
        cv=3,
        n_jobs=1,
    )

    raise AssertionError("Expected ValueError")

except ValueError:
    print("Invalid parameter grid handled correctly.")


# ============================================================
# TEST 10: INVALID METHOD HANDLING
# ============================================================

print("\nTEST 10: Invalid Method")

try:
    tune_model(
        model=LogisticRegression(),
        X=X_cls,
        y=y_cls,
        param_grid={"C": [1.0]},
        method="invalid",
        cv=3,
    )

    raise AssertionError("Expected ValueError")

except ValueError:
    print("Invalid method handled correctly.")


print("\n========== ALL HYPERPARAMETER TUNING TESTS COMPLETED ==========")

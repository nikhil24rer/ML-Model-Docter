
"""
Model Engine
------------
Trains and compares candidate machine learning models.

Responsibilities:
1. Detect target problem type.
2. Validate dataset inputs.
3. Split data into training and testing sets.
4. Select candidate models dynamically.
5. Build leakage-safe preprocessing pipelines.
6. Select models using cross-validation on training data.
7. Evaluate the selected model on an untouched holdout test set.
8. Compare candidates against a dummy baseline.
9. Return trained pipelines and internal evaluation data.

Evaluation strategy:
- Classification: StratifiedKFold cross-validation.
- Regression: KFold cross-validation.
- Model selection uses mean CV scores only.
- Holdout test data is reserved for final evaluation.
- Preprocessing is fitted independently inside each CV fold.
"""

import math
import warnings

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from ml.target_detector import detect_target_type
from ml.preprocessing import build_preprocessor
from ml.model_registry import (
    get_model_registry,
    create_model,
)
from ml.model_selector import select_models


# ==================================================
# 1. TARGET TYPE DETECTION
# ==================================================

def detect_problem_type(y, user_choice=None):
    """
    Detect and normalize target problem type.

    Supported outputs:
        binary_classification
        multiclass_classification
        regression
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    result = detect_target_type(
        y,
        target_name=y.name or "target",
        user_choice=user_choice,
    )

    detected_type = result["problem_type"]

    if detected_type in ("invalid", "unsupported"):
        raise ValueError(
            result.get("reason", "Invalid target column.")
        )

    if result.get("requires_confirmation"):
        raise ValueError(
            "Target type is ambiguous. Please confirm whether "
            "this is a classification or regression problem."
        )

    if detected_type == "regression":
        return "regression"

    if detected_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        number_of_classes = y.dropna().nunique()

        if number_of_classes == 2:
            return "binary_classification"

        if number_of_classes > 2:
            return "multiclass_classification"

    raise ValueError(
        f"Unable to determine target type: {detected_type}"
    )


# ==================================================
# 2. PROBLEM TYPE NORMALIZATION
# ==================================================

def normalize_problem_type(problem_type, y=None):
    """Normalize classification aliases."""

    if problem_type == "regression":
        return "regression"

    if problem_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        if y is None:
            return problem_type

        number_of_classes = y.nunique()

        if number_of_classes < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        if number_of_classes == 2:
            return "binary_classification"

        return "multiclass_classification"

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


# ==================================================
# 3. PREPROCESSING COMPATIBILITY FUNCTIONS
# ==================================================

def build_model_preprocessor(X, algorithm="random_forest"):
    """Build preprocessing using the dedicated preprocessing module."""

    return build_preprocessor(
        X,
        algorithm=algorithm,
    )


def build_preprocessor_for_engine(
    X,
    algorithm="random_forest",
):
    """Backward-compatible function name."""

    return build_model_preprocessor(X, algorithm)


# ==================================================
# 4. DENSE ARRAY CONVERSION
# ==================================================

def convert_to_dense(X):
    """Gaussian Naive Bayes requires dense input."""

    if sparse.issparse(X):
        return X.toarray()

    return X


# ==================================================
# 5. BUILD MODEL PIPELINE
# ==================================================

def build_model_pipeline(
    X_train,
    model_name,
    problem_type,
):
    """
    Build a complete preprocessing + model pipeline.

    Preprocessing is fitted only when pipeline.fit() is called.
    """

    registry = get_model_registry(problem_type)

    if model_name not in registry:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model_kwargs = {}

    # Prevent KNN from requesting more neighbors than
    # available training samples.
    if model_name in ("knn", "knn_regressor"):
        model_kwargs["n_neighbors"] = min(
            5,
            len(X_train),
        )

    model = create_model(
        model_name,
        problem_type,
        **model_kwargs,
    )

    preprocessing = build_preprocessor(
        X_train,
        algorithm=model_name,
    )

    steps = [
        ("preprocessing", preprocessing),
    ]

    # GaussianNB does not support sparse matrices.
    if model_name == "naive_bayes":
        steps.append((
            "to_dense",
            FunctionTransformer(
                convert_to_dense,
                validate=False,
            ),
        ))

    steps.append(("model", model))

    return Pipeline(steps)


# ==================================================
# 6. CREATE DUMMY BASELINE
# ==================================================

def build_baseline_pipeline(X_train, problem_type):
    """Create a simple baseline for comparison."""

    if problem_type == "regression":
        baseline_model = DummyRegressor(
            strategy="mean"
        )
    else:
        baseline_model = DummyClassifier(
            strategy="most_frequent"
        )

    preprocessing = build_preprocessor(
        X_train,
        algorithm="random_forest",
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", baseline_model),
    ])


# ==================================================
# 7. CLASSIFICATION METRICS
# ==================================================

def evaluate_classification(y_true, predictions):
    """Calculate classification evaluation metrics."""

    return {
        "accuracy": round(
            float(accuracy_score(y_true, predictions)),
            4,
        ),

        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, predictions)),
            4,
        ),

        "precision": round(
            float(precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )),
            4,
        ),

        "recall": round(
            float(recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )),
            4,
        ),

        "f1": round(
            float(f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )),
            4,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
    }


# ==================================================
# 8. REGRESSION METRICS
# ==================================================

def evaluate_regression(y_true, predictions):
    """Calculate regression evaluation metrics."""

    return {
        "r2": round(
            float(r2_score(y_true, predictions)),
            4,
        ),

        "mae": round(
            float(mean_absolute_error(y_true, predictions)),
            4,
        ),

        "rmse": round(
            float(np.sqrt(
                mean_squared_error(y_true, predictions)
            )),
            4,
        ),
    }


# ==================================================
# 9. INPUT VALIDATION
# ==================================================

def validate_training_data(X, y, test_size):
    """Validate features, target and split configuration."""

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if len(X) != len(y):
        raise ValueError(
            "Features and target must have the same number of rows."
        )

    if len(X) < 10:
        raise ValueError(
            "At least 10 valid rows are required for holdout evaluation."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "At least one feature column is required."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature column names are not supported."
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values. "
            "Remove missing target rows before training."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    # Remove completely empty feature columns.
    empty_columns = X.columns[X.isna().all()].tolist()

    if empty_columns:
        X = X.drop(columns=empty_columns)

    if X.shape[1] == 0:
        raise ValueError(
            "All feature columns contain missing values."
        )

    # Reset indexes to guarantee positional alignment.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, empty_columns


# ==================================================
# 10. CROSS-VALIDATION CONFIGURATION
# ==================================================

def build_cv_strategy(y_train, problem_type, max_folds=5):
    """
    Build a suitable cross-validation strategy.

    Classification uses StratifiedKFold.
    Regression uses KFold.

    Number of folds is reduced when the dataset is small.
    """

    if problem_type != "regression":

        class_counts = y_train.value_counts()

        min_class_count = int(class_counts.min())

        if min_class_count < 2:
            raise ValueError(
                "Cross-validation requires at least 2 training "
                "samples for every target class."
            )

        n_splits = min(
            max_folds,
            min_class_count,
        )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

    n_splits = min(
        max_folds,
        len(y_train),
    )

    if n_splits < 2:
        raise ValueError(
            "At least 2 training samples are required for cross-validation."
        )

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )


def get_cv_scoring(problem_type):
    """Return the primary cross-validation scoring metric."""

    if problem_type == "regression":
        return "r2"

    return "f1_macro"


# ==================================================
# 11. MAIN MODEL COMPARISON FUNCTION
# ==================================================

def compare_models(
    X,
    y,
    problem_type=None,
    max_models=6,
    test_size=0.20,
    user_choice=None,
):
    """
    Train and compare candidate machine learning models.

    Model selection:
        Uses cross-validation scores from training data only.

    Final evaluation:
        Uses the untouched holdout test set.

    Returns:
        Model results, selected model, baseline metrics,
        CV scores and internal fitted objects.
    """

    # ----------------------------------------------
    # A. Validate inputs
    # ----------------------------------------------

    X, y, removed_columns = validate_training_data(
        X,
        y,
        test_size,
    )

    if not isinstance(max_models, int) or max_models <= 0:
        raise ValueError(
            "max_models must be a positive integer."
        )

    # ----------------------------------------------
    # B. Detect problem type
    # ----------------------------------------------

    if problem_type is None:
        problem_type = detect_problem_type(
            y,
            user_choice=user_choice,
        )
    else:
        problem_type = normalize_problem_type(
            problem_type,
            y,
        )

    problem_type = normalize_problem_type(
        problem_type,
        y,
    )

    is_classification = problem_type != "regression"

    # ----------------------------------------------
    # C. Dataset information
    # ----------------------------------------------

    n_samples = len(X)
    n_features = X.shape[1]

    n_classes = (
        int(y.nunique())
        if is_classification
        else None
    )

    is_imbalanced = False
    imbalance_ratio = None

    if is_classification:

        class_counts = y.value_counts()

        if class_counts.min() < 2:
            raise ValueError(
                "Every target class needs at least 2 samples "
                "for stratified holdout evaluation."
            )

        imbalance_ratio = (
            float(class_counts.max()) /
            float(class_counts.min())
        )

        is_imbalanced = imbalance_ratio >= 3

    # ----------------------------------------------
    # D. Validate train-test split
    # ----------------------------------------------

    if is_classification:

        test_rows = math.ceil(n_samples * test_size)
        train_rows = n_samples - test_rows

        if test_rows < n_classes or train_rows < n_classes:
            raise ValueError(
                "The dataset is too small for a stratified split "
                "with all target classes represented in both sets. "
                "Increase dataset size or adjust test_size."
            )

        stratify = y

    else:
        stratify = None

    # ----------------------------------------------
    # E. Split dataset
    # ----------------------------------------------

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

    except ValueError as exc:
        raise ValueError(
            "Unable to create a valid train-test split. "
            "Check dataset size and class distribution. "
            f"Details: {exc}"
        ) from exc

    # ----------------------------------------------
    # F. Configure cross-validation
    # ----------------------------------------------

    cv_strategy = build_cv_strategy(
        y_train,
        problem_type,
    )

    cv_folds = cv_strategy.get_n_splits()

    cv_scoring = get_cv_scoring(problem_type)

    # ----------------------------------------------
    # G. Select candidate models
    # ----------------------------------------------

    selection = select_models(
        n_samples=n_samples,
        n_features=n_features,
        problem_type=problem_type,
        n_classes=n_classes,
        is_imbalanced=is_imbalanced,
        max_models=max_models,
    )

    candidate_names = selection["selected_models"]

    if not candidate_names:
        raise ValueError(
            "No suitable candidate models were selected."
        )

    registry = get_model_registry(problem_type)

    # ----------------------------------------------
    # H. Cross-validate candidate models
    # ----------------------------------------------

    results = []
    cv_results = {}
    trained_models = {}

    for model_name in candidate_names:

        display_name = registry[model_name]["name"]

        try:

            pipeline = build_model_pipeline(
                X_train,
                model_name,
                problem_type,
            )

            # Each CV fold gets its own cloned pipeline.
            # Preprocessing is fitted only on that fold's training data.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fold_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=cv_scoring,
                    n_jobs=1,
                    error_score="raise",
                )

            if not np.all(np.isfinite(fold_scores)):
                raise ValueError(
                    "Cross-validation produced non-finite scores."
                )

            mean_cv_score = float(np.mean(fold_scores))
            std_cv_score = float(np.std(fold_scores))

            # --------------------------------------
            # I. Fit candidate on complete train set
            # --------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            train_predictions = pipeline.predict(X_train)
            test_predictions = pipeline.predict(X_test)

            result = {
                "model": display_name,
                "model_id": model_name,
                "status": "success",

                "cv_folds": int(cv_folds),
                "cv_scoring": cv_scoring,
                "cv_mean_score": round(mean_cv_score, 4),
                "cv_std_score": round(std_cv_score, 4),
                "cv_scores": [
                    round(float(score), 4)
                    for score in fold_scores
                ],

                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }

            if is_classification:

                train_metrics = evaluate_classification(
                    y_train,
                    train_predictions,
                )

                test_metrics = evaluate_classification(
                    y_test,
                    test_predictions,
                )

                result["train_accuracy"] = (
                    train_metrics["accuracy"]
                )

                result["train_metrics"] = train_metrics

                result.update(test_metrics)

                # Legacy field: holdout macro F1.
                # This is NOT used for model selection.
                result["score"] = test_metrics["f1"]

            else:

                train_metrics = evaluate_regression(
                    y_train,
                    train_predictions,
                )

                test_metrics = evaluate_regression(
                    y_test,
                    test_predictions,
                )

                result["train_r2"] = train_metrics["r2"]

                result["train_metrics"] = train_metrics

                result.update(test_metrics)

                # Legacy field: holdout R2.
                # This is NOT used for model selection.
                result["score"] = test_metrics["r2"]

            results.append(result)

            cv_results[model_name] = {
                "mean_score": round(mean_cv_score, 4),
                "std_score": round(std_cv_score, 4),
                "fold_scores": [
                    round(float(score), 4)
                    for score in fold_scores
                ],
            }

            trained_models[model_name] = pipeline

        except Exception as exc:

            results.append({
                "model": display_name,
                "model_id": model_name,
                "status": "failed",
                "error": str(exc),
            })

    # ----------------------------------------------
    # J. Identify best candidate using CV only
    # ----------------------------------------------

    valid_results = [
        result
        for result in results
        if result.get("status") == "success"
        and np.isfinite(result.get("cv_mean_score", np.nan))
    ]

    if not valid_results:
        raise ValueError(
            "All candidate models failed. "
            "Inspect individual model errors."
        )

    best = max(
        valid_results,
        key=lambda item: item["cv_mean_score"],
    )

    best_model_id = best["model_id"]
    best_model_pipeline = trained_models[best_model_id]

    # ----------------------------------------------
    # K. Dummy baseline
    # ----------------------------------------------

    baseline_pipeline = build_baseline_pipeline(
        X_train,
        problem_type,
    )

    baseline_pipeline.fit(
        X_train,
        y_train,
    )

    baseline_train_predictions = baseline_pipeline.predict(X_train)
    baseline_test_predictions = baseline_pipeline.predict(X_test)

    if is_classification:

        baseline_train_metrics = evaluate_classification(
            y_train,
            baseline_train_predictions,
        )

        baseline_result = evaluate_classification(
            y_test,
            baseline_test_predictions,
        )

        primary_metric = "macro F1"

        baseline_result.pop(
            "confusion_matrix",
            None,
        )

        baseline_train_metrics.pop(
            "confusion_matrix",
            None,
        )

    else:

        baseline_train_metrics = evaluate_regression(
            y_train,
            baseline_train_predictions,
        )

        baseline_result = evaluate_regression(
            y_test,
            baseline_test_predictions,
        )

        primary_metric = "R2"

    baseline_result["train_metrics"] = baseline_train_metrics

    # ----------------------------------------------
    # L. Final response
    # ----------------------------------------------

    return {
        "problem_type": problem_type,

        "evaluation_method": (
            f"{int((1 - test_size) * 100)}/"
            f"{int(test_size * 100)} holdout "
            f"+ {cv_folds}-fold cross-validation"
        ),

        "primary_metric": primary_metric,

        "selection_metric": cv_scoring,

        "selection_method": "cross_validation",

        "cv_folds": int(cv_folds),

        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),

        "dataset_info": {
            "total_rows": int(n_samples),
            "total_features": int(n_features),
            "number_of_classes": n_classes,
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": (
                round(imbalance_ratio, 4)
                if imbalance_ratio is not None
                else None
            ),
            "removed_empty_columns": removed_columns,
        },

        "models": results,

        "cv_results": cv_results,

        "best_model": best["model"],
        "best_model_id": best_model_id,

        # best_score now represents the CV mean.
        "best_score": best["cv_mean_score"],

        "best_cv_std": best["cv_std_score"],

        "best_test_score": best["score"],

        "baseline": baseline_result,

        "selection": selection,

        "successful_models": len(valid_results),
        "failed_models": len(results) - len(valid_results),

        "score_interpretation": (
            "The best candidate was selected using cross-validation "
            "scores from training data only. The holdout test set was "
            "not used for model selection and is reserved for final "
            "evaluation. The final test score is still an estimate "
            "based on one holdout split and may vary with new data."
        ),

        # Internal objects: never directly JSON-serialize.
        "_trained_models": trained_models,

        "_baseline_pipeline": baseline_pipeline,

        "_best_model_pipeline": best_model_pipeline,

        # Held-out data for explainability and detailed evaluation.
        # These are internal and must not be returned by Flask.
        "_X_train": X_train,
        "_X_test": X_test,
        "_y_train": y_train,
        "_y_test": y_test,

        "_baseline_train_metrics": baseline_train_metrics,
    }

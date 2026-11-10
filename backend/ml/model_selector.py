
"""
Model Selector
--------------
Selects suitable candidate ML algorithms based on dataset
characteristics.

This module does NOT train models or guarantee the best model.
It only recommends candidates for subsequent evaluation.
"""

from ml.model_registry import get_model_registry


# --------------------------------------------------
# 1. DEFAULT CONFIGURATION
# --------------------------------------------------

DEFAULT_MAX_MODELS = 6
MIN_SAMPLES_FOR_KNN = 5
HIGH_DIMENSION_RATIO = 10


# --------------------------------------------------
# 2. VALIDATE DATASET INFORMATION
# --------------------------------------------------

def validate_dataset_info(
    n_samples,
    n_features,
    problem_type,
    n_classes=None
):
    """Validate dataset metadata before model selection."""

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer.")

    supported_types = {
        "classification",
        "binary_classification",
        "multiclass_classification",
        "regression",
    }

    if problem_type not in supported_types:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    if problem_type != "regression":
        if n_classes is not None:
            if not isinstance(n_classes, int) or n_classes < 2:
                raise ValueError(
                    "n_classes must be an integer greater than or equal to 2."
                )


# --------------------------------------------------
# 3. SELECT CLASSIFICATION MODELS
# --------------------------------------------------

def select_classification_models(
    n_samples,
    n_features,
    is_imbalanced=False,
):
    """
    Select classification candidates.

    Dataset size, feature dimensionality and imbalance
    influence the candidate list.
    """

    candidates = []

    # Linear baseline: useful for many classification datasets.
    candidates.append("logistic_regression")

    # Tree-based models.
    candidates.append("random_forest")
    candidates.append("extra_trees")

    # Additional non-linear candidate.
    candidates.append("gradient_boosting")

    # KNN can be unreliable when samples are extremely few.
    if n_samples >= MIN_SAMPLES_FOR_KNN:
        candidates.append("knn")

    # Naive Bayes is a useful probabilistic baseline.
    candidates.append("naive_bayes")

    # SVM can be expensive on very large datasets.
    if n_samples <= 20000:
        candidates.append("svm")

    # High-dimensional data: retain linear and regularized
    # style candidates; avoid adding more complex candidates
    # merely because the feature count is high.
    if n_features >= n_samples * HIGH_DIMENSION_RATIO:
        candidates = [
            "logistic_regression",
            "naive_bayes",
            "svm",
            "random_forest",
        ]

    # Imbalance is not solved by choosing a model alone.
    # Keep candidates and let the training/evaluation stage
    # handle class weights, resampling and suitable metrics.

    return candidates


# --------------------------------------------------
# 4. SELECT REGRESSION MODELS
# --------------------------------------------------

def select_regression_models(
    n_samples,
    n_features,
):
    """Select regression candidates."""

    candidates = [
        "linear_regression",
        "ridge",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
    ]

    if n_samples >= MIN_SAMPLES_FOR_KNN:
        candidates.append("knn_regressor")

    # SVR can be expensive for large datasets.
    if n_samples <= 10000:
        candidates.append("svr")

    if n_features >= n_samples * HIGH_DIMENSION_RATIO:
        candidates = [
            "linear_regression",
            "ridge",
            "random_forest_regressor",
        ]

    return candidates


# --------------------------------------------------
# 5. MAIN MODEL SELECTION FUNCTION
# --------------------------------------------------

def select_models(
    n_samples,
    n_features,
    problem_type,
    n_classes=None,
    is_imbalanced=False,
    max_models=DEFAULT_MAX_MODELS,
):
    """
    Select candidate models based on dataset characteristics.

    Parameters:
        n_samples: Number of rows in the dataset.
        n_features: Number of usable input features.
        problem_type: Classification or regression.
        n_classes: Number of target classes, if classification.
        is_imbalanced: Whether the target classes are imbalanced.
        max_models: Maximum number of candidates to return.

    Returns:
        Dictionary containing candidate model identifiers,
        model metadata and selection reasons.
    """

    validate_dataset_info(
        n_samples=n_samples,
        n_features=n_features,
        problem_type=problem_type,
        n_classes=n_classes,
    )

    if not isinstance(max_models, int) or max_models <= 0:
        raise ValueError("max_models must be a positive integer.")

    if problem_type == "regression":
        candidates = select_regression_models(
            n_samples=n_samples,
            n_features=n_features,
        )
    else:
        candidates = select_classification_models(
            n_samples=n_samples,
            n_features=n_features,
            is_imbalanced=is_imbalanced,
        )

    # Keep only models available in the registry.
    registry = get_model_registry(problem_type)

    candidates = [
        name for name in candidates
        if name in registry
    ]

    # Remove duplicates while preserving order.
    candidates = list(dict.fromkeys(candidates))

    # Apply requested limit.
    selected_models = candidates[:max_models]

    # Build selection reasons.
    reasons = []

    if n_samples < 100:
        reasons.append(
            "Small dataset: model performance may be unstable."
        )
    elif n_samples > 10000:
        reasons.append(
            "Large dataset: computational cost should be considered."
        )

    if n_features > n_samples:
        reasons.append(
            "More features than samples: overfitting risk is elevated."
        )

    if is_imbalanced and problem_type != "regression":
        reasons.append(
            "Class imbalance detected: use stratified evaluation "
            "and imbalance-aware metrics."
        )

    if n_features >= n_samples * HIGH_DIMENSION_RATIO:
        reasons.append(
            "Very high feature-to-sample ratio: simpler candidates prioritized."
        )

    if not reasons:
        reasons.append(
            "Candidates selected using dataset size and feature count."
        )

    model_details = {
        name: {
            "name": registry[name]["name"],
            "description": registry[name]["description"],
            "requires_scaling": registry[name]["requires_scaling"],
            "supports_probability": registry[name]["supports_probability"],
        }
        for name in selected_models
    }

    return {
        "problem_type": problem_type,
        "dataset_info": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "is_imbalanced": (
                is_imbalanced if problem_type != "regression" else False
            ),
        },
        "selected_models": selected_models,
        "model_details": model_details,
        "selection_reasons": reasons,
        "selection_note": (
            "These are candidate models, not guaranteed best models. "
            "Train and evaluate them using leakage-safe validation "
            "before selecting a final model."
        ),
    }

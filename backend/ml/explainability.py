
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# ============================================================
# 1. VALIDATION HELPERS
# ============================================================

def validate_feature_names(feature_names):
    """Validate and normalize feature names."""

    if feature_names is None:
        return None

    names = [str(name) for name in feature_names]

    if not names:
        raise ValueError("Feature names cannot be empty.")

    return names


def validate_feature_importance(values, feature_names=None):
    """Validate importance values and feature names."""

    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) == 0:
        raise ValueError("Importance values cannot be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Importance values must be finite.")

    names = validate_feature_names(feature_names)

    if names is not None and len(names) != len(values):
        raise ValueError(
            "Feature names and importance values must have equal lengths."
        )

    if names is None:
        names = [f"feature_{i}" for i in range(len(values))]

    return values, names


# ============================================================
# 2. BUILD FEATURE IMPORTANCE SUMMARY
# ============================================================

def build_importance_summary(
    values,
    feature_names=None,
    method="unknown",
    top_n=10,
):
    """
    Build a sorted feature importance summary.

    Supports positive and negative values, such as model coefficients.
    """

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    values, names = validate_feature_importance(
        values,
        feature_names,
    )

    records = []

    for name, value in zip(names, values):
        records.append({
            "feature": name,
            "importance": float(value),
            "absolute_importance": float(abs(value)),
        })

    records.sort(
        key=lambda item: item["absolute_importance"],
        reverse=True,
    )

    return {
        "method": method,
        "total_features": len(records),
        "top_features": records[:top_n],
        "all_features": records,
        "interpretation": (
            "Importance represents model-specific feature influence. "
            "It does not establish causation."
        ),
    }


# ============================================================
# 3. EXTRACT FINAL ESTIMATOR FROM PIPELINE
# ============================================================

def get_final_estimator(model):
    """Return the final estimator from a sklearn Pipeline."""

    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


# ============================================================
# 4. EXTRACT TRANSFORMED FEATURE NAMES
# ============================================================

def get_transformed_feature_names(model, X=None):
    """
    Get feature names after preprocessing.

    Supports sklearn pipelines containing a ColumnTransformer.
    Falls back to original input columns when possible.
    """

    if isinstance(model, Pipeline):

        if len(model.steps) > 1:

            preprocessing = model[:-1]

            try:
                return list(
                    preprocessing.get_feature_names_out()
                )
            except (AttributeError, ValueError, TypeError):
                pass

    if X is not None:

        if hasattr(X, "columns"):
            return [str(column) for column in X.columns]

        return [
            f"feature_{i}"
            for i in range(np.asarray(X).shape[1])
        ]

    return None


# ============================================================
# 5. NATIVE FEATURE IMPORTANCE
# ============================================================

def get_native_feature_importance(
    model,
    feature_names=None,
    top_n=10,
):
    """
    Extract feature_importances_ from supported estimators.
    """

    estimator = get_final_estimator(model)

    if not hasattr(estimator, "feature_importances_"):
        return {
            "available": False,
            "reason": (
                "This estimator does not expose native "
                "feature_importances_."
            ),
        }

    values = estimator.feature_importances_

    summary = build_importance_summary(
        values,
        feature_names,
        method="native_feature_importance",
        top_n=top_n,
    )

    return {
        "available": True,
        **summary,
    }


# ============================================================
# 6. LINEAR MODEL COEFFICIENTS
# ============================================================

def get_model_coefficients(
    model,
    feature_names=None,
    top_n=10,
):
    """
    Extract coefficients from linear estimators.

    For multiclass models, importance is aggregated using
    mean absolute coefficient across classes.
    """

    estimator = get_final_estimator(model)

    if not hasattr(estimator, "coef_"):
        return {
            "available": False,
            "reason": "This estimator does not expose coef_.",
        }

    coefficients = np.asarray(estimator.coef_, dtype=float)

    if coefficients.ndim == 1:
        aggregated = coefficients

    elif coefficients.ndim == 2:
        aggregated = np.mean(
            np.abs(coefficients),
            axis=0,
        )

    else:
        return {
            "available": False,
            "reason": "Unsupported coefficient dimensions.",
        }

    summary = build_importance_summary(
        aggregated,
        feature_names,
        method="model_coefficients",
        top_n=top_n,
    )

    return {
        "available": True,
        **summary,
    }


# ============================================================
# 7. PERMUTATION IMPORTANCE
# ============================================================

def get_permutation_importance(
    model,
    X,
    y,
    scoring=None,
    n_repeats=5,
    random_state=42,
    top_n=10,
):
    """
    Calculate model-agnostic permutation importance.

    The supplied model should already be fitted.
    Prefer a separate validation set instead of training data.
    """

    if X is None or y is None:
        raise ValueError("X and y are required.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    if len(X) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer.")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    if hasattr(X, "columns"):
        feature_names = [str(col) for col in X.columns]
    else:
        feature_names = [
            f"feature_{i}"
            for i in range(np.asarray(X).shape[1])
        ]

    summary = build_importance_summary(
        result.importances_mean,
        feature_names,
        method="permutation_importance",
        top_n=top_n,
    )

    uncertainty = {
        name: float(std)
        for name, std in zip(
            feature_names,
            result.importances_std,
        )
    }

    for item in summary["all_features"]:
        item["importance_std"] = uncertainty[item["feature"]]

    for item in summary["top_features"]:
        item["importance_std"] = uncertainty[item["feature"]]

    return {
        "available": True,
        **summary,
    }


# ============================================================
# 8. INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

def explain_prediction(
    model,
    X,
    row_index=0,
):
    """
    Explain an individual prediction using model output.

    This reports the prediction and available probability.
    It does not claim causal feature attribution.
    """

    if X is None or len(X) == 0:
        raise ValueError("Input data cannot be empty.")

    if not isinstance(row_index, int):
        raise ValueError("row_index must be an integer.")

    if row_index < 0 or row_index >= len(X):
        raise IndexError("row_index is outside the input data.")

    if hasattr(X, "iloc"):
        sample = X.iloc[[row_index]]
        feature_values = X.iloc[row_index].to_dict()
    else:
        sample_array = np.asarray(X)
        sample = sample_array[[row_index]]
        feature_values = {
            f"feature_{i}": value
            for i, value in enumerate(sample_array[row_index])
        }

    prediction = model.predict(sample)[0]

    result = {
        "prediction": (
            prediction.item()
            if isinstance(prediction, np.generic)
            else prediction
        ),
        "input_features": feature_values,
        "probabilities": None,
        "predicted_class_probability": None,
        "explanation_method": "prediction_and_input_summary",
        "limitation": (
            "This output shows the prediction and input values. "
            "It does not identify each feature's causal contribution."
        ),
    }

    if hasattr(model, "predict_proba"):

        try:
            probabilities = model.predict_proba(sample)[0]

            classes = getattr(model, "classes_", None)

            if classes is None:
                estimator = get_final_estimator(model)
                classes = getattr(estimator, "classes_", None)

            if classes is not None:

                result["probabilities"] = {
                    str(label): float(probability)
                    for label, probability in zip(
                        classes,
                        probabilities,
                    )
                }

                matching = np.where(classes == prediction)[0]

                if len(matching):
                    result["predicted_class_probability"] = float(
                        probabilities[matching[0]]
                    )

        except (ValueError, AttributeError, TypeError):
            pass

    return result


# ============================================================
# 9. AUTOMATIC EXPLAINABILITY REPORT
# ============================================================

def explain_model(
    model,
    X=None,
    y=None,
    feature_names=None,
    top_n=10,
    include_permutation=False,
    scoring=None,
    n_repeats=5,
):
    """
    Generate a combined explainability report.

    Native importance and coefficients are used when available.
    Permutation importance is optional.
    """

    native_importance = get_native_feature_importance(
        model,
        feature_names=feature_names,
        top_n=top_n,
    )

    coefficients = get_model_coefficients(
        model,
        feature_names=feature_names,
        top_n=top_n,
    )

    permutation = None

    if include_permutation:

        if X is None or y is None:
            raise ValueError(
                "X and y are required for permutation importance."
            )

        permutation = get_permutation_importance(
            model=model,
            X=X,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            top_n=top_n,
        )

    available_methods = []

    if native_importance["available"]:
        available_methods.append("native_feature_importance")

    if coefficients["available"]:
        available_methods.append("model_coefficients")

    if permutation and permutation["available"]:
        available_methods.append("permutation_importance")

    return {
        "available": bool(available_methods),
        "methods": available_methods,
        "native_importance": native_importance,
        "coefficients": coefficients,
        "permutation_importance": permutation,
        "limitations": [
            "Feature importance does not prove causation.",
            "Native importance values are model-specific.",
            "Correlated features can share or distort importance.",
            "Permutation importance depends on the evaluation dataset.",
        ],
    }

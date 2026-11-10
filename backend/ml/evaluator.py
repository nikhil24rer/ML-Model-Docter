
"""
Model Evaluator
---------------
Evaluates trained classification and regression models.

Responsibilities:
1. Calculate classification metrics.
2. Calculate regression metrics.
3. Generate class-wise performance.
4. Compare training and testing performance.
5. Identify possible overfitting and underfitting.
6. Compare model performance against a baseline.
7. Generate a structured evaluation report.

Important:
- Evaluation does not train models.
- Overfitting indicators are diagnostic signals, not guarantees.
- Holdout metrics are only as reliable as the evaluation dataset.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


# ==================================================
# 1. VALIDATE INPUTS
# ==================================================

def validate_predictions(y_true, predictions):
    """
    Validate actual and predicted target values.
    """

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    if y_true.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "Actual and predicted values must be one-dimensional."
        )

    if len(y_true) != len(predictions):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation requires at least one sample."
        )

    if pd.isna(y_true).any():
        raise ValueError(
            "Actual target contains missing values."
        )

    if pd.isna(predictions).any():
        raise ValueError(
            "Predictions contain missing values."
        )

    return y_true, predictions


# ==================================================
# 2. CLASSIFICATION EVALUATION
# ==================================================

def evaluate_classification_model(
    y_true,
    predictions,
):
    """
    Calculate classification evaluation metrics.

    Returns:
        accuracy
        balanced_accuracy
        precision
        recall
        f1
        confusion_matrix
        class_wise_metrics
    """

    y_true, predictions = validate_predictions(
        y_true,
        predictions,
    )

    labels = np.unique(
        np.concatenate([y_true, predictions])
    )

    class_report = classification_report(
        y_true,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    class_wise_metrics = {}

    for label in labels:

        label_key = str(label)

        if label_key in class_report:
            details = class_report[label_key]

            class_wise_metrics[label_key] = {
                "precision": round(
                    float(details["precision"]), 4
                ),
                "recall": round(
                    float(details["recall"]), 4
                ),
                "f1": round(
                    float(details["f1-score"]), 4
                ),
                "support": int(details["support"]),
            }

    return {
        "accuracy": round(
            float(accuracy_score(y_true, predictions)), 4
        ),

        "balanced_accuracy": round(
            float(balanced_accuracy_score(
                y_true,
                predictions,
            )), 4
        ),

        "precision": round(
            float(precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )), 4
        ),

        "recall": round(
            float(recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )), 4
        ),

        "f1": round(
            float(f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )), 4
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=labels,
        ).tolist(),

        "class_labels": [
            str(label) for label in labels
        ],

        "class_wise_metrics": class_wise_metrics,

        "total_samples": int(len(y_true)),
    }


# ==================================================
# 3. REGRESSION EVALUATION
# ==================================================

def evaluate_regression_model(
    y_true,
    predictions,
):
    """
    Calculate regression evaluation metrics.

    Returns:
        r2
        mae
        rmse
        median_absolute_error
        residual statistics
    """

    y_true, predictions = validate_predictions(
        y_true,
        predictions,
    )

    try:
        r2 = float(r2_score(y_true, predictions))
    except ValueError:
        r2 = float("nan")

    mae = float(
        mean_absolute_error(y_true, predictions)
    )

    rmse = float(
        np.sqrt(mean_squared_error(
            y_true,
            predictions,
        ))
    )

    median_ae = float(
        median_absolute_error(
            y_true,
            predictions,
        )
    )

    residuals = y_true - predictions

    return {
        "r2": round(r2, 4) if np.isfinite(r2) else None,

        "mae": round(mae, 4),

        "rmse": round(rmse, 4),

        "median_absolute_error": round(
            median_ae, 4
        ),

        "mean_residual": round(
            float(np.mean(residuals)), 4
        ),

        "residual_std": round(
            float(np.std(residuals)), 4
        ),

        "total_samples": int(len(y_true)),
    }


# ==================================================
# 4. COMPARE TRAINING AND TEST PERFORMANCE
# ==================================================

def compare_train_test_performance(
    train_metrics,
    test_metrics,
    problem_type,
    gap_threshold=0.10,
):
    """
    Compare training and testing performance.

    For classification:
        Uses macro F1.

    For regression:
        Uses R2.

    A large positive train-test gap can indicate overfitting.
    """

    if not 0 <= gap_threshold <= 1:
        raise ValueError(
            "gap_threshold must be between 0 and 1."
        )

    if problem_type == "regression":
        metric_name = "r2"
    elif problem_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        metric_name = "f1"
    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    if (
        metric_name not in train_metrics
        or metric_name not in test_metrics
    ):
        raise ValueError(
            f"Missing required metric: {metric_name}"
        )

    train_score = train_metrics[metric_name]
    test_score = test_metrics[metric_name]

    if train_score is None or test_score is None:
        raise ValueError(
            "Training and testing scores cannot be None."
        )

    train_score = float(train_score)
    test_score = float(test_score)

    if not np.isfinite(train_score) or not np.isfinite(test_score):
        raise ValueError(
            "Training and testing scores must be finite."
        )

    performance_gap = train_score - test_score

    overfitting_suspected = (
        performance_gap >= gap_threshold
    )

    return {
        "metric": metric_name,

        "train_score": round(train_score, 4),

        "test_score": round(test_score, 4),

        "performance_gap": round(
            performance_gap, 4
        ),

        "gap_threshold": gap_threshold,

        "overfitting_suspected": bool(
            overfitting_suspected
        ),

        "status": (
            "possible_overfitting"
            if overfitting_suspected
            else "no_large_generalization_gap_detected"
        ),

        "interpretation": (
            "Training performance is substantially higher "
            "than testing performance. Possible overfitting "
            "should be investigated."
            if overfitting_suspected
            else
            "No large train-test performance gap was detected. "
            "This does not guarantee that the model generalizes well."
        ),
    }


# ==================================================
# 5. BASELINE COMPARISON
# ==================================================

def compare_with_baseline(
    model_metrics,
    baseline_metrics,
    problem_type,
):
    """
    Compare model performance against a dummy baseline.

    Classification:
        Higher macro F1 is better.

    Regression:
        Higher R2 is better.
        Lower MAE and RMSE are better.
    """

    if problem_type == "regression":

        metric_names = ["r2", "mae", "rmse"]

    elif problem_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):

        metric_names = [
            "accuracy",
            "balanced_accuracy",
            "f1",
        ]

    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    comparison = {}

    for metric in metric_names:

        if metric not in model_metrics or metric not in baseline_metrics:
            continue

        model_value = model_metrics[metric]
        baseline_value = baseline_metrics[metric]

        if model_value is None or baseline_value is None:
            continue

        model_value = float(model_value)
        baseline_value = float(baseline_value)

        if (
            not np.isfinite(model_value)
            or not np.isfinite(baseline_value)
        ):
            continue

        difference = model_value - baseline_value

        comparison[metric] = {
            "model": round(model_value, 4),
            "baseline": round(baseline_value, 4),
            "difference": round(difference, 4),
            "improved": (
                difference < 0
                if metric in ("mae", "rmse")
                else difference > 0
            ),
        }

    primary_metric = (
        "r2"
        if problem_type == "regression"
        else "f1"
    )

    primary_comparison = comparison.get(primary_metric)

    if primary_comparison is None:
        status = "not_available"
    elif primary_comparison["improved"]:
        status = "better_than_baseline"
    elif primary_comparison["difference"] == 0:
        status = "equal_to_baseline"
    else:
        status = "worse_than_baseline"

    return {
        "primary_metric": primary_metric,
        "status": status,
        "metrics": comparison,
    }


# ==================================================
# 6. UNDERFITTING INDICATOR
# ==================================================

def detect_underfitting(
    train_metrics,
    baseline_metrics,
    problem_type,
):
    """
    Detect a possible underfitting signal.

    A model is flagged when its training score does not
    outperform the corresponding dummy baseline score.

    This is a warning signal, not a definitive diagnosis.
    """

    if problem_type == "regression":
        metric = "r2"
    elif problem_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        metric = "f1"
    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    if metric not in train_metrics or metric not in baseline_metrics:
        raise ValueError(
            f"Missing metric required for underfitting check: {metric}"
        )

    train_score = train_metrics[metric]
    baseline_score = baseline_metrics[metric]

    if train_score is None or baseline_score is None:
        raise ValueError(
            "Training and baseline scores cannot be None."
        )

    train_score = float(train_score)
    baseline_score = float(baseline_score)

    if not np.isfinite(train_score) or not np.isfinite(baseline_score):
        raise ValueError(
            "Training and baseline scores must be finite."
        )

    suspected = train_score <= baseline_score

    return {
        "metric": metric,
        "train_score": round(train_score, 4),
        "baseline_score": round(baseline_score, 4),
        "underfitting_suspected": bool(suspected),
        "interpretation": (
            "Training performance does not exceed the dummy "
            "baseline. Investigate feature quality, model "
            "capacity and preprocessing."
            if suspected
            else
            "Training performance exceeds the dummy baseline. "
            "This alone does not rule out underfitting."
        ),
    }


# ==================================================
# 7. BUILD COMPLETE EVALUATION REPORT
# ==================================================


def build_evaluation_report(
    train_metrics,
    test_metrics,
    baseline_train_metrics,
    baseline_test_metrics,
    problem_type,
    gap_threshold=0.10,
):
    """
    Combine evaluation results into one structured report.

    Baseline training metrics are used for underfitting checks.
    Baseline testing metrics are used for test-set comparison.
    """

    generalization = compare_train_test_performance(
        train_metrics,
        test_metrics,
        problem_type,
        gap_threshold=gap_threshold,
    )

    baseline_comparison = compare_with_baseline(
        test_metrics,
        baseline_test_metrics,
        problem_type,
    )

    underfitting = detect_underfitting(
        train_metrics,
        baseline_train_metrics,
        problem_type,
    )

    warnings = []

    if generalization["overfitting_suspected"]:
        warnings.append(
            "Possible overfitting: training performance "
            "is substantially higher than testing performance."
        )

    if underfitting["underfitting_suspected"]:
        warnings.append(
            "Possible underfitting: training performance "
            "does not exceed dummy baseline performance."
        )

    if baseline_comparison["status"] == "worse_than_baseline":
        warnings.append(
            "The model performs worse than the dummy baseline "
            "on the primary evaluation metric."
        )

    if not warnings:
        warnings.append(
            "No major warning was identified by the current "
            "evaluation checks. Further validation is still recommended."
        )

    return {
        "problem_type": problem_type,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_train_metrics": baseline_train_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "generalization": generalization,
        "baseline_comparison": baseline_comparison,
        "underfitting": underfitting,
        "warnings": warnings,
        "evaluation_note": (
            "These checks provide diagnostic indicators, not "
            "proof of model quality. Use cross-validation and "
            "an independent test dataset for stronger evidence."
        ),
    }


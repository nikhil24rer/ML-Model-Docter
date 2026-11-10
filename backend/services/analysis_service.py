
import os
import re
from datetime import datetime, timezone

import pandas as pd

from ml.data_profiler import profile_dataset
from ml.data_cleaner import clean_dataset
from ml.target_detector import detect_target_type
from ml.model_engine import compare_models
from ml.diagnosis import diagnose_model
from ml.recommendations import generate_recommendation
from ml.explainability import (
    explain_model,
    get_transformed_feature_names,
)
from ml.evaluator import build_evaluation_report
from ml.report_generator import (
    generate_report,
    export_json_report,
    export_html_report,
)


# ============================================================
# 1. HELPERS
# ============================================================

def validate_analysis_input(df, target_column):
    """Validate the uploaded DataFrame and selected target."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataset must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    if not target_column:
        raise ValueError("Please select a target column.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset must contain at least one feature and one target."
        )


def safe_filename(value):
    """Create a safe filename for generated reports."""

    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)

    return value.strip("_") or "dataset"


def get_public_comparison(comparison):
    """
    Remove fitted sklearn objects before returning or exporting
    the comparison dictionary.
    """

    return {
        key: value
        for key, value in comparison.items()
        if not key.startswith("_")
    }



def get_selected_trained_model(comparison):
    """
    Retrieve the fitted pipeline for the selected best model.
    Models are stored using their model_id, not display name.
    """

    trained_models = comparison.get("_trained_models", {})

    if not isinstance(trained_models, dict):
        return None

    # First priority: exact model ID
    best_model_id = comparison.get("best_model_id")

    if best_model_id and best_model_id in trained_models:
        return trained_models[best_model_id]

    # Fallback: display name matching
    best_model_name = comparison.get("best_model")

    if best_model_name:
        normalized_name = str(best_model_name).strip().lower()

        for model_id, model in trained_models.items():
            if str(model_id).strip().lower() == normalized_name:
                return model

    return None


def generate_evaluation_report(comparison):
    """
    Build a detailed evaluation report for the selected model.
    """

    best_model_id = comparison.get("best_model_id")

    selected_result = next(
        (
            item
            for item in comparison.get("models", [])
            if item.get("model_id") == best_model_id
            and item.get("status") == "success"
        ),
        None,
    )

    if selected_result is None:
        return {
            "available": False,
            "reason": "Selected model evaluation metrics are unavailable.",
        }

    problem_type = comparison["problem_type"]

    if problem_type == "regression":
        metric_keys = ["r2", "mae", "rmse"]
    else:
        metric_keys = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
        ]

    train_metrics = selected_result.get("train_metrics")
    baseline_train_metrics = comparison.get(
        "_baseline_train_metrics"
    )
    baseline = comparison.get("baseline", {})

    if not train_metrics or not baseline_train_metrics:
        return {
            "available": False,
            "reason": "Required training metrics are unavailable.",
        }

    test_metrics = {
        key: selected_result[key]
        for key in metric_keys
        if key in selected_result
    }

    baseline_test_metrics = {
        key: baseline[key]
        for key in metric_keys
        if key in baseline
    }

    try:
        report = build_evaluation_report(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            baseline_train_metrics=baseline_train_metrics,
            baseline_test_metrics=baseline_test_metrics,
            problem_type=problem_type,
        )

        report["available"] = True
        report["selected_model"] = comparison.get("best_model")

        return report

    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def generate_explainability_report(comparison, X):
    """
    Generate explainability using the selected trained model.

    Native importance and coefficients are extracted from the
    fitted model. Permutation importance uses untouched holdout data.
    """

    report = {
        "available": False,
        "methods": [],
        "native_importance": {
            "available": False,
            "reason": "No selected trained model was available.",
        },
        "coefficients": {
            "available": False,
            "reason": "No selected trained model was available.",
        },
        "permutation_importance": None,
        "limitations": [
            "Feature importance does not prove causation.",
            "Native importance values are model-specific.",
            "Correlated features can share or distort importance.",
            "Permutation importance depends on evaluation data.",
        ],
    }

    feature_importance = None
    selected_model = get_selected_trained_model(comparison)

    if selected_model is None:
        return report, feature_importance

    try:
        transformed_names = get_transformed_feature_names(
            selected_model,
            X=X,
        )

        X_test = comparison.get("_X_test")
        y_test = comparison.get("_y_test")

        include_permutation = (
            X_test is not None
            and y_test is not None
            and len(X_test) > 0
        )

        report = explain_model(
            model=selected_model,
            feature_names=transformed_names,
            X=X_test if include_permutation else None,
            y=y_test if include_permutation else None,
            top_n=10,
            include_permutation=include_permutation,
            n_repeats=5,
        )

        # Prefer native importance, then coefficients,
        # then permutation importance.
        if report["native_importance"]["available"]:
            feature_importance = (
                report["native_importance"]["all_features"]
            )

        elif report["coefficients"]["available"]:
            feature_importance = (
                report["coefficients"]["all_features"]
            )

        elif (
            report.get("permutation_importance")
            and report["permutation_importance"].get("available")
        ):
            feature_importance = (
                report["permutation_importance"]["all_features"]
            )

    except Exception as exc:
        report["available"] = False
        report["error"] = str(exc)

    return report, feature_importance

# ============================================================
# 2. MAIN ANALYSIS SERVICE
# ============================================================

def analyze_dataset(
    df,
    target_column,
    user_choice=None,
    max_models=6,
    test_size=0.20,
    generate_reports=True,
    report_directory="reports",
):
    """
    Run the complete dataset analysis workflow.

    Parameters
    ----------
    df : pandas.DataFrame
        Uploaded dataset.

    target_column : str
        Selected prediction target.

    user_choice : str, optional
        User confirmation for ambiguous target types.

    max_models : int
        Maximum number of candidate models.

    test_size : float
        Holdout test fraction.

    generate_reports : bool
        Whether to export JSON and HTML reports.

    report_directory : str
        Directory in which reports are saved.

    Returns
    -------
    dict
        Unified analysis result.
    """

    started_at = datetime.now(timezone.utc)

    # --------------------------------------------------------
    # STEP 1: Validate input
    # --------------------------------------------------------

    validate_analysis_input(df, target_column)

    original_df = df.copy()

    # --------------------------------------------------------
    # STEP 2: Profile original dataset
    # --------------------------------------------------------

    dataset_profile = profile_dataset(original_df)

    # --------------------------------------------------------
    # STEP 3: Detect target type
    # --------------------------------------------------------

    target_info = detect_target_type(
        original_df[target_column].dropna(),
        target_name=target_column,
        user_choice=user_choice,
    )

    if (
        target_info.get("problem_type") in ("ambiguous", "invalid")
        or target_info.get("requires_confirmation", False)
    ):

        return {
            "status": (
                "target_confirmation_required"
                if target_info.get("problem_type") == "ambiguous"
                else "error"
            ),
            "message": (
                "Please confirm whether the target represents "
                "classification or regression."
                if target_info.get("problem_type") == "ambiguous"
                else "The selected target cannot be used for this analysis."
            ),
            "target_info": target_info,
            "dataset_profile": dataset_profile,
        }

    # --------------------------------------------------------
    # STEP 4: Clean dataset for downloadable export
    # --------------------------------------------------------

    cleaned_df, cleaning_report = clean_dataset(
        original_df,
        target_column=target_column,
    )

    # --------------------------------------------------------
    # STEP 5: Prepare model data
    # --------------------------------------------------------

    # Model evaluation uses original feature values.
    # Preprocessing is performed inside model pipelines.

    model_df = original_df.dropna(
        subset=[target_column]
    ).copy()

    model_df = model_df.drop_duplicates().reset_index(drop=True)

    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    # --------------------------------------------------------
    # STEP 6: Add analysis metadata to profile
    # --------------------------------------------------------

    dataset_profile["original_missing_values"] = int(
        original_df.isna().sum().sum()
    )

    dataset_profile["original_duplicate_rows"] = int(
        original_df.duplicated().sum()
    )

    dataset_profile["target_missing_values"] = int(
        original_df[target_column].isna().sum()
    )

    dataset_profile["target_distribution"] = {
        str(key): int(value)
        for key, value in y.value_counts().to_dict().items()
    }

    # --------------------------------------------------------
    # STEP 7: Train and compare models
    # --------------------------------------------------------

    comparison = compare_models(
        X,
        y,
        problem_type=None,
        max_models=max_models,
        test_size=test_size,
        user_choice=user_choice,
    )
    
        # --------------------------------------------------------
    # STEP 8: Generate detailed evaluation
    # --------------------------------------------------------

    evaluation_report = generate_evaluation_report(comparison)

    # --------------------------------------------------------
    # STEP 8: Generate explainability
    # --------------------------------------------------------

    explainability_report, feature_importance = (
        generate_explainability_report(
            comparison=comparison,
            X=X,
        )
    )

    # --------------------------------------------------------
    # STEP 9: Generate recommendations
    # --------------------------------------------------------

    recommendation = generate_recommendation(
        comparison,
        dataset_profile,
    )

    # --------------------------------------------------------
    # STEP 10: Generate diagnosis
    # --------------------------------------------------------

        # --------------------------------------------------------
    # STEP 10: Generate diagnosis
    # --------------------------------------------------------

    diagnosis_report = diagnose_model(
        dataset_profile=dataset_profile,
        target_info=target_info,
        feature_importance=feature_importance,
        model_name=comparison.get("best_model"),
        evaluation_report=evaluation_report,
    )
    # --------------------------------------------------------
    # STEP 11: Prepare public model comparison
    # --------------------------------------------------------

    public_comparison = get_public_comparison(comparison)

    public_comparison["explainability"] = explainability_report
    public_comparison["evaluation"] = evaluation_report

    # --------------------------------------------------------
    # STEP 12: Generate reports
    # --------------------------------------------------------

    report = generate_report(
        dataset_profile=dataset_profile,
        target_info=target_info,
        model_results=public_comparison,
        diagnosis_report=diagnosis_report,
        metadata={
            "target_column": target_column,
            "created_at": started_at.isoformat(),
            "application": "ML Model Doctor",
        },
    )

    report_paths = {}

    if generate_reports:

        os.makedirs(report_directory, exist_ok=True)

        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        base_name = safe_filename(target_column)

        json_path = os.path.join(
            report_directory,
            f"{base_name}_{timestamp}.json",
        )

        html_path = os.path.join(
            report_directory,
            f"{base_name}_{timestamp}.html",
        )

        report_paths["json"] = export_json_report(
            report,
            json_path,
        )

        report_paths["html"] = export_html_report(
            report,
            html_path,
        )

    # --------------------------------------------------------
    # STEP 13: Build final response
    # --------------------------------------------------------

    completed_at = datetime.now(timezone.utc)

    return {
        "status": "success",

        "dataset": {
            "rows": int(model_df.shape[0]),
            "columns": int(model_df.shape[1]),
            "features": int(X.shape[1]),
            "target": target_column,
        },

        "problem": {
            "type": comparison["problem_type"],
            "target": target_column,
            "target_analysis": target_info,
        },

        "models": comparison["models"],

        "model_comparison": public_comparison,

                "evaluation": {
            "method": comparison["evaluation_method"],
            "primary_metric": comparison["primary_metric"],
            "selection_metric": comparison["selection_metric"],
            "selection_method": comparison["selection_method"],
            "cv_folds": comparison["cv_folds"],
            "train_rows": comparison["train_rows"],
            "test_rows": comparison["test_rows"],
            "best_cv_score": comparison["best_score"],
            "best_cv_std": comparison["best_cv_std"],
            "best_test_score": comparison["best_test_score"],
            "baseline": comparison["baseline"],
            "details": evaluation_report,
            "score_interpretation": comparison.get(
                "score_interpretation"
            ),
        },

        "recommendation": recommendation,

        "diagnosis": diagnosis_report,

        "explainability": explainability_report,

        "profile": dataset_profile,

        "cleaning_report": cleaning_report,

        "cleaned_dataframe": cleaned_df,

        "report": report,

        "report_paths": report_paths,

        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(
                (completed_at - started_at).total_seconds(),
                3,
            ),
        },

        # Internal fitted objects.
        # Never directly jsonify these objects.
        "_trained_models": comparison.get("_trained_models", {}),

        "_baseline_pipeline": comparison.get(
            "_baseline_pipeline"
        ),
    }

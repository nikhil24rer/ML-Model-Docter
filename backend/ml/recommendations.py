

def generate_recommendation(comparison_result, profile):

    best_model = comparison_result.get("best_model")
    problem_type = comparison_result.get("problem_type")
    metric = comparison_result.get("primary_metric")

    diagnosis = []
    suggestions = []

    # --------------------------------------------------
    # 1. Dataset quality checks
    # --------------------------------------------------

    missing_values = profile.get("original_missing_values", 0)

    if missing_values > 0:
        diagnosis.append(
            f"{missing_values} missing cells were found "
            "in the original dataset."
        )
        suggestions.append(
            "Review the cleaning report and inspect columns "
            "with a high percentage of missing values."
        )

    duplicate_rows = profile.get("original_duplicate_rows", 0)

    if duplicate_rows > 0:
        diagnosis.append(
            f"{duplicate_rows} exact duplicate rows were found."
        )
        suggestions.append(
            "Verify whether duplicate records are legitimate "
            "before removing them permanently."
        )

    missing_targets = profile.get("target_missing_values", 0)

    if missing_targets > 0:
        diagnosis.append(
            f"{missing_targets} rows with missing target labels "
            "were excluded from model training."
        )

    # --------------------------------------------------
    # 2. Identify successfully evaluated models
    # --------------------------------------------------

    valid_models = [
        model
        for model in comparison_result.get("models", [])
        if model.get("status") == "success"
    ]

    if not valid_models:
        return {
            "recommended_model": None,
            "reason": "No candidate model completed successfully.",
            "diagnosis": diagnosis,
            "suggestions": [
                "Review individual model errors and dataset quality.",
                "Check whether the dataset has enough valid samples."
            ],
        }

    # --------------------------------------------------
    # 3. Classification-specific checks
    # --------------------------------------------------

    classification_types = (
        "classification",
        "binary_classification",
        "multiclass_classification",
    )

    if problem_type in classification_types:

        dataset_info = comparison_result.get("dataset_info", {})

        if dataset_info.get("is_imbalanced", False):
            diagnosis.append(
                "The target classes are imbalanced. "
                "Accuracy alone may give a misleading picture."
            )

            suggestions.append(
                "Review macro F1, balanced accuracy and "
                "the confusion matrix."
            )

        # Compare holdout F1 with holdout baseline F1.
        baseline = comparison_result.get("baseline", {})

        baseline_f1 = baseline.get("f1")
        best_test_score = comparison_result.get("best_test_score")

        if (
            baseline_f1 is not None
            and best_test_score is not None
            and best_test_score <= baseline_f1
        ):
            diagnosis.append(
                "The selected model did not outperform the "
                "dummy baseline on holdout macro F1."
            )

            suggestions.append(
                "Investigate class imbalance, feature quality "
                "and whether the target is predictable."
            )

    # --------------------------------------------------
    # 4. Regression-specific checks
    # --------------------------------------------------

    elif problem_type == "regression":

        baseline = comparison_result.get("baseline", {})

        baseline_r2 = baseline.get("r2")
        best_test_score = comparison_result.get("best_test_score")

        if (
            baseline_r2 is not None
            and best_test_score is not None
            and best_test_score <= baseline_r2
        ):
            diagnosis.append(
                "The selected model did not outperform the "
                "dummy baseline on holdout R2."
            )

            suggestions.append(
                "Review feature quality, target relationships "
                "and regression errors."
            )

    # --------------------------------------------------
    # 5. Selected model information
    # --------------------------------------------------

    best_cv_score = comparison_result.get("best_score")
    best_test_score = comparison_result.get("best_test_score")

    diagnosis.append(
        f"{best_model} was selected using "
        "cross-validation performance."
    )

    suggestions.append(
        "Treat the holdout result as an initial estimate. "
        "Validate on another representative dataset before deployment."
    )

    return {
        "recommended_model": best_model,

        "reason": (
            f"{best_model} achieved the strongest cross-validation "
            f"score ({best_cv_score}) among the evaluated candidates. "
            f"Its holdout test score was {best_test_score}. "
            "Future performance is not guaranteed."
        ),

        "selection_metric": comparison_result.get("selection_metric"),

        "cv_score": best_cv_score,

        "test_score": best_test_score,

        "diagnosis": diagnosis,

        "suggestions": suggestions,
    }



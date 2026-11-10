
from typing import Any, Dict, List, Optional


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """Safely convert values to float."""
    try:
        if value is None:
            return None

        result = float(value)

        if result != result:  # NaN check
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def add_finding(
    findings: List[Dict[str, Any]],
    code: str,
    title: str,
    severity: str,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
):
    """Create a structured diagnostic finding."""

    findings.append({
        "code": code,
        "title": title,
        "severity": severity,
        "message": message,
        "evidence": evidence or {},
        "recommendation": recommendation,
    })


# ============================================================
# 2. OVERFITTING DIAGNOSIS
# ============================================================

def diagnose_generalization(evaluation_report, findings):

    generalization = evaluation_report.get("generalization", {})

    if not isinstance(generalization, dict):
        return

    suspected = generalization.get("overfitting_suspected")

    if suspected is True:

        add_finding(
            findings,
            code="possible_overfitting",
            title="Possible Overfitting",
            severity="warning",
            message=(
                "Training performance is substantially higher than "
                "testing performance. The model may not generalize "
                "well to unseen data."
            ),
            evidence={
                "metric": generalization.get("metric"),
                "train_score": generalization.get("train_score"),
                "test_score": generalization.get("test_score"),
                "performance_gap": generalization.get("performance_gap"),
                "gap_threshold": generalization.get("gap_threshold"),
            },
            recommendation=(
                "Investigate model complexity, possible data leakage, "
                "dataset size and validation strategy. Consider "
                "regularization or simpler models."
            ),
        )

    elif suspected is False:

        add_finding(
            findings,
            code="no_large_generalization_gap",
            title="No Large Generalization Gap Detected",
            severity="info",
            message=(
                "No large train-test performance gap was identified. "
                "This does not guarantee good real-world performance."
            ),
            evidence={
                "metric": generalization.get("metric"),
                "train_score": generalization.get("train_score"),
                "test_score": generalization.get("test_score"),
                "performance_gap": generalization.get("performance_gap"),
            },
            recommendation=(
                "Validate performance with cross-validation and "
                "independent data where possible."
            ),
        )


# ============================================================
# 3. BASELINE DIAGNOSIS
# ============================================================

def diagnose_baseline(evaluation_report, findings):

    comparison = evaluation_report.get("baseline_comparison", {})

    if not isinstance(comparison, dict):
        return

    status = comparison.get("status")

    if status == "better_than_baseline":

        add_finding(
            findings,
            code="better_than_baseline",
            title="Model Outperforms Dummy Baseline",
            severity="info",
            message=(
                "The model performs better than the dummy baseline "
                "on the primary evaluation metric."
            ),
            evidence=comparison,
            recommendation=(
                "Continue evaluating with cross-validation and "
                "check whether the improvement is consistent."
            ),
        )

    elif status == "not_better_than_baseline":

        add_finding(
            findings,
            code="not_better_than_baseline",
            title="Model Does Not Beat Dummy Baseline",
            severity="warning",
            message=(
                "The model has not demonstrated improvement over "
                "the dummy baseline on the primary metric."
            ),
            evidence=comparison,
            recommendation=(
                "Review target quality, feature relevance, preprocessing "
                "and model selection."
            ),
        )


# ============================================================
# 4. UNDERFITTING DIAGNOSIS
# ============================================================

def diagnose_underfitting(evaluation_report, findings):

    underfitting = evaluation_report.get("underfitting", {})

    if not isinstance(underfitting, dict):
        return

    suspected = underfitting.get("underfitting_suspected")

    if suspected is True:

        add_finding(
            findings,
            code="possible_underfitting",
            title="Possible Underfitting",
            severity="warning",
            message=(
                "Training performance does not exceed dummy baseline "
                "training performance. The model may not be learning "
                "useful patterns."
            ),
            evidence={
                "metric": underfitting.get("metric"),
                "train_score": underfitting.get("train_score"),
                "baseline_score": underfitting.get("baseline_score"),
            },
            recommendation=(
                "Check feature quality, preprocessing and model capacity. "
                "Consider testing a more expressive model."
            ),
        )


# ============================================================
# 5. DATA QUALITY DIAGNOSIS
# ============================================================

def diagnose_data_quality(dataset_profile, findings):

    if not isinstance(dataset_profile, dict):
        return

    # Missing values

    missing_percentage = dataset_profile.get("missing_percentage", {})
    missing_values = dataset_profile.get("missing_values", {})

    if isinstance(missing_percentage, dict):

        affected_columns = {
            column: percentage
            for column, percentage in missing_percentage.items()
            if (safe_float(percentage) or 0) > 0
        }

        if affected_columns:

            add_finding(
                findings,
                code="missing_values_detected",
                title="Missing Values Detected",
                severity="warning",
                message=(
                    "Some dataset columns contain missing values."
                ),
                evidence={
                    "columns": affected_columns,
                    "missing_values": missing_values,
                    "columns_with_missing": dataset_profile.get(
                        "columns_with_missing", []
                    ),
                },
                recommendation=(
                    "Inspect missing-value patterns and select "
                    "an appropriate imputation strategy."
                ),
            )

    # Duplicate rows

    duplicate_rows = dataset_profile.get("duplicate_rows", 0)

    if (safe_float(duplicate_rows) or 0) > 0:

        add_finding(
            findings,
            code="duplicate_rows_detected",
            title="Duplicate Rows Detected",
            severity="warning",
            message="The dataset contains duplicate rows.",
            evidence={"duplicate_rows": duplicate_rows},
            recommendation=(
                "Inspect duplicates and determine whether they "
                "represent valid observations."
            ),
        )

    # Constant columns

    constant_columns = dataset_profile.get("constant_columns", [])

    if constant_columns:

        add_finding(
            findings,
            code="constant_features_detected",
            title="Constant Columns Detected",
            severity="warning",
            message=(
                "Some columns contain only one distinct value "
                "and may not provide predictive information."
            ),
            evidence={"columns": constant_columns},
            recommendation=(
                "Consider removing constant columns before training."
            ),
        )

    # Outliers

    outlier_rows = dataset_profile.get("outlier_rows", 0)

    if (safe_float(outlier_rows) or 0) > 0:

        add_finding(
            findings,
            code="possible_outliers_detected",
            title="Possible Outliers Detected",
            severity="info",
            message=(
                "The profiler identified possible statistical outliers. "
                "These values are not automatically data errors."
            ),
            evidence={
                "outlier_rows": outlier_rows,
                "outlier_values": dataset_profile.get("outlier_values"),
                "outlier_details": dataset_profile.get("outlier_details"),
            },
            recommendation=(
                "Inspect outliers using domain knowledge before "
                "removing or transforming them."
            ),
        )

    # Identifier columns

    identifier_candidates = dataset_profile.get(
        "identifier_candidates", []
    )

    if identifier_candidates:

        add_finding(
            findings,
            code="possible_identifier_features",
            title="Possible Identifier Columns",
            severity="warning",
            message=(
                "Some columns appear to contain identifiers and "
                "may not provide useful predictive information."
            ),
            evidence={"columns": identifier_candidates},
            recommendation=(
                "Check whether these columns represent genuine "
                "identifiers and remove them if they are not useful."
            ),
        )

    # High cardinality

    high_cardinality = dataset_profile.get(
        "high_cardinality_columns", []
    )

    if high_cardinality:

        add_finding(
            findings,
            code="high_cardinality_features",
            title="High-Cardinality Features Detected",
            severity="info",
            message=(
                "Some categorical columns contain many distinct values."
            ),
            evidence={"columns": high_cardinality},
            recommendation=(
                "Inspect these columns and consider grouping rare "
                "categories or using an appropriate encoding strategy."
            ),
        )

    # Profiler warnings

    profiler_warnings = dataset_profile.get("warnings", [])

    if profiler_warnings:

        add_finding(
            findings,
            code="profiler_warnings",
            title="Dataset Profiler Warnings",
            severity="info",
            message=(
                "The dataset profiler reported additional observations."
            ),
            evidence={"warnings": profiler_warnings},
            recommendation=(
                "Review the profiler warnings before finalizing "
                "dataset preparation."
            ),
        )


# ============================================================
# 6. TARGET AND CLASS IMBALANCE DIAGNOSIS
# ============================================================

def diagnose_target(target_info, findings):

    if not isinstance(target_info, dict):
        return

    problem_type = target_info.get("problem_type")

    if problem_type in ("ambiguous", "invalid", "unsupported"):

        add_finding(
            findings,
            code="target_type_requires_attention",
            title="Target Type Needs Attention",
            severity="warning",
            message=(
                "The target detector could not confidently establish "
                "a supported problem type."
            ),
            evidence={
                "problem_type": problem_type,
                "suggested_problem_type": target_info.get(
                    "suggested_problem_type"
                ),
                "reason": target_info.get("reason"),
                "warnings": target_info.get("warnings", []),
            },
            recommendation=(
                "Verify the target column and select classification "
                "or regression when its meaning is clear."
            ),
        )

    class_analysis = target_info.get("class_analysis", {})

    if not isinstance(class_analysis, dict):
        return

    if class_analysis.get("is_imbalanced") is True:

        add_finding(
            findings,
            code="class_imbalance_detected",
            title="Class Imbalance Detected",
            severity="warning",
            message=(
                "The target has an uneven class distribution. "
                "Accuracy alone may hide poor minority-class performance."
            ),
            evidence=class_analysis,
            recommendation=(
                "Review per-class precision, recall, balanced accuracy "
                "and the confusion matrix. Consider class weights "
                "or resampling when appropriate."
            ),
        )


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

def summarize_feature_importance(feature_importance):

    if feature_importance is None:
        return None

    records = []

    if isinstance(feature_importance, dict):

        records = [
            {"feature": name, "importance": value}
            for name, value in feature_importance.items()
        ]

    elif isinstance(feature_importance, list):

        records = feature_importance

    elif hasattr(feature_importance, "to_dict"):

        try:
            records = feature_importance.to_dict(orient="records")
        except TypeError:
            return None

    normalized = []

    for record in records:

        if not isinstance(record, dict):
            continue

        feature = record.get(
            "feature",
            record.get("Feature", record.get("name")),
        )

        importance = safe_float(
            record.get("importance", record.get("Importance"))
        )

        if feature is not None and importance is not None:

            normalized.append({
                "feature": str(feature),
                "importance": importance,
            })

    normalized.sort(
        key=lambda item: abs(item["importance"]),
        reverse=True,
    )

    return {
        "available": bool(normalized),
        "top_features": normalized[:10],
        "total_features": len(normalized),
        "interpretation": (
            "Importance indicates relative model reliance, "
            "not causation."
        ),
    }


# ============================================================
# 8. MAIN DIAGNOSIS ENGINE
# ============================================================

def diagnose_model(
    evaluation_report: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    target_info: Optional[Dict[str, Any]] = None,
    feature_importance=None,
    model_name: Optional[str] = None,
):

    evaluation_report = evaluation_report or {}
    dataset_profile = dataset_profile or {}
    target_info = target_info or {}

    findings = []

    # Run all diagnostic checks

    diagnose_generalization(evaluation_report, findings)

    diagnose_baseline(evaluation_report, findings)

    diagnose_underfitting(evaluation_report, findings)

    diagnose_data_quality(dataset_profile, findings)

    diagnose_target(target_info, findings)

    # Feature importance

    importance_summary = summarize_feature_importance(
        feature_importance
    )

    if importance_summary and importance_summary["available"]:

        add_finding(
            findings,
            code="feature_importance_available",
            title="Feature Importance Available",
            severity="info",
            message=(
                "Feature importance information is available "
                "for the evaluated model."
            ),
            evidence=importance_summary,
            recommendation=(
                "Review important features with domain knowledge. "
                "Do not interpret importance as causation."
            ),
        )

    # No findings case

    if not findings:

        add_finding(
            findings,
            code="insufficient_diagnostic_evidence",
            title="Insufficient Diagnostic Evidence",
            severity="info",
            message=(
                "No diagnostic issues were identified from the "
                "information supplied. This does not prove that "
                "the model or dataset is problem-free."
            ),
            recommendation=(
                "Provide evaluation results, dataset profiling "
                "and target analysis for a more complete diagnosis."
            ),
        )

    # Sort findings by severity

    severity_order = {
        "critical": 0,
        "warning": 1,
        "info": 2,
    }

    findings.sort(
        key=lambda item: severity_order.get(item["severity"], 3)
    )

    # Summary

    summary = {
        "total_findings": len(findings),

        "critical": sum(
            item["severity"] == "critical"
            for item in findings
        ),

        "warnings": sum(
            item["severity"] == "warning"
            for item in findings
        ),

        "information": sum(
            item["severity"] == "info"
            for item in findings
        ),
    }

    status = (
        "issues_found"
        if summary["critical"] > 0 or summary["warnings"] > 0
        else "no_major_issues_detected"
    )

    return {
        "model_name": model_name,
        "status": status,
        "summary": summary,
        "findings": findings,
        "feature_importance": importance_summary,
        "limitations": [
            "Diagnosis depends on the completeness of supplied inputs.",
            "Heuristic findings are signals for investigation, not proof.",
            "Feature importance does not establish causation.",
            "A small train-test gap does not guarantee real-world performance.",
        ],
    }


# ============================================================
# 9. BACKWARD COMPATIBILITY
# ============================================================

generate_diagnosis = diagnose_model

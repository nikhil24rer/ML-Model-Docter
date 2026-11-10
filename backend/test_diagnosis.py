
from ml.diagnosis import (
    diagnose_model,
    generate_diagnosis,
    summarize_feature_importance,
)


def get_finding_codes(report):
    return [item["code"] for item in report["findings"]]


# ============================================================
# TEST 1: OVERFITTING DIAGNOSIS
# ============================================================

print("\nTEST 1: Overfitting Diagnosis")

evaluation = {
    "generalization": {
        "metric": "f1",
        "train_score": 0.98,
        "test_score": 0.65,
        "performance_gap": 0.33,
        "gap_threshold": 0.10,
        "overfitting_suspected": True,
    }
}

report = diagnose_model(evaluation_report=evaluation)

assert "possible_overfitting" in get_finding_codes(report)
print(report["findings"])
print("Overfitting test passed.")


# ============================================================
# TEST 2: BASELINE DIAGNOSIS
# ============================================================

print("\nTEST 2: Baseline Diagnosis")

evaluation = {
    "baseline_comparison": {
        "primary_metric": "f1",
        "status": "not_better_than_baseline",
        "metrics": {
            "f1": {
                "model": 0.40,
                "baseline": 0.50,
                "difference": -0.10,
                "improved": False,
            }
        },
    }
}

report = diagnose_model(evaluation_report=evaluation)

assert "not_better_than_baseline" in get_finding_codes(report)
print("Baseline test passed.")


# ============================================================
# TEST 3: UNDERFITTING DIAGNOSIS
# ============================================================

print("\nTEST 3: Underfitting Diagnosis")

evaluation = {
    "underfitting": {
        "metric": "f1",
        "train_score": 0.30,
        "baseline_score": 0.40,
        "underfitting_suspected": True,
    }
}

report = diagnose_model(evaluation_report=evaluation)

assert "possible_underfitting" in get_finding_codes(report)
print("Underfitting test passed.")


# ============================================================
# TEST 4: DATA QUALITY DIAGNOSIS
# ============================================================

print("\nTEST 4: Data Quality Diagnosis")

profile = {
    "rows": 100,
    "columns": 6,
    "missing_values": {
        "Age": 10,
        "Salary": 5,
    },
    "missing_percentage": {
        "Age": 10.0,
        "Salary": 5.0,
    },
    "columns_with_missing": ["Age", "Salary"],
    "duplicate_rows": 8,
    "constant_columns": ["Constant_Column"],
    "outlier_rows": 4,
    "outlier_values": 6,
    "outlier_details": {},
    "identifier_candidates": ["Customer_ID"],
    "high_cardinality_columns": ["City"],
    "warnings": ["Some columns contain missing values."],
}

report = diagnose_model(dataset_profile=profile)

codes = get_finding_codes(report)

expected_codes = [
    "missing_values_detected",
    "duplicate_rows_detected",
    "constant_features_detected",
    "possible_outliers_detected",
    "possible_identifier_features",
    "high_cardinality_features",
    "profiler_warnings",
]

for code in expected_codes:
    assert code in codes, f"Missing diagnostic finding: {code}"

print("Detected findings:", codes)
print("Data quality test passed.")


# ============================================================
# TEST 5: CLASS IMBALANCE
# ============================================================

print("\nTEST 5: Class Imbalance Diagnosis")

target_info = {
    "problem_type": "binary_classification",
    "class_analysis": {
        "class_counts": {
            "0": 90,
            "1": 10,
        },
        "imbalance_ratio": 9.0,
        "is_imbalanced": True,
    },
}

report = diagnose_model(target_info=target_info)

assert "class_imbalance_detected" in get_finding_codes(report)

print("Class imbalance test passed.")


# ============================================================
# TEST 6: TARGET TYPE NEEDS ATTENTION
# ============================================================

print("\nTEST 6: Target Type Diagnosis")

target_info = {
    "problem_type": "ambiguous",
    "suggested_problem_type": "classification",
    "reason": "Integer target has a small number of unique values.",
    "warnings": ["User confirmation required."],
}

report = diagnose_model(target_info=target_info)

assert "target_type_requires_attention" in get_finding_codes(report)

print("Target diagnosis test passed.")


# ============================================================
# TEST 7: FEATURE IMPORTANCE
# ============================================================

print("\nTEST 7: Feature Importance Summary")

importance = {
    "Age": 0.25,
    "Salary": 0.60,
    "Experience": 0.15,
}

summary = summarize_feature_importance(importance)

assert summary["available"] is True
assert summary["top_features"][0]["feature"] == "Salary"
assert summary["total_features"] == 3

report = diagnose_model(feature_importance=importance)

assert "feature_importance_available" in get_finding_codes(report)

print(summary)
print("Feature importance test passed.")


# ============================================================
# TEST 8: COMPLETE DIAGNOSIS
# ============================================================

print("\nTEST 8: Complete Diagnosis")

evaluation = {
    "generalization": {
        "metric": "f1",
        "train_score": 0.95,
        "test_score": 0.60,
        "performance_gap": 0.35,
        "gap_threshold": 0.10,
        "overfitting_suspected": True,
    },
    "baseline_comparison": {
        "primary_metric": "f1",
        "status": "better_than_baseline",
    },
    "underfitting": {
        "metric": "f1",
        "train_score": 0.95,
        "baseline_score": 0.40,
        "underfitting_suspected": False,
    },
}

target_info = {
    "problem_type": "binary_classification",
    "class_analysis": {
        "is_imbalanced": True,
        "imbalance_ratio": 8.0,
    },
}

report = diagnose_model(
    evaluation_report=evaluation,
    dataset_profile=profile,
    target_info=target_info,
    feature_importance=importance,
    model_name="Random Forest",
)

assert report["model_name"] == "Random Forest"
assert report["status"] == "issues_found"
assert report["summary"]["total_findings"] > 0
assert report["summary"]["warnings"] > 0

print("Model:", report["model_name"])
print("Status:", report["status"])
print("Summary:", report["summary"])

for finding in report["findings"]:
    print(
        finding["severity"],
        "|",
        finding["title"],
        "|",
        finding["recommendation"],
    )

print("Complete diagnosis test passed.")


# ============================================================
# TEST 9: EMPTY INPUT
# ============================================================

print("\nTEST 9: Empty Input Handling")

report = diagnose_model()

assert report["status"] == "no_major_issues_detected"
assert "insufficient_diagnostic_evidence" in get_finding_codes(report)

print("Empty input test passed.")


# ============================================================
# TEST 10: ALIAS COMPATIBILITY
# ============================================================

print("\nTEST 10: Function Alias")

report = generate_diagnosis(
    model_name="Test Model"
)

assert report["model_name"] == "Test Model"

print("Alias compatibility test passed.")


print("\n========== ALL DIAGNOSIS TESTS COMPLETED ==========")


import json
import tempfile
from pathlib import Path

from ml.report_generator import (
    generate_report,
    build_executive_summary,
    export_json_report,
    build_html_report,
    export_html_report,
    build_dashboard_summary,
    save_report,
)


# ============================================================
# TEST 1: BASIC REPORT GENERATION
# ============================================================

print("\nTEST 1: Basic Report Generation")

report = generate_report(
    dataset_profile={"rows": 100, "columns": 5},
    target_info={"problem_type": "classification"},
)

assert report["report_version"] == "1.0"
assert report["dataset_profile"]["rows"] == 100
assert report["target_analysis"]["problem_type"] == "classification"

print("Basic report generation passed.")


# ============================================================
# TEST 2: EXECUTIVE SUMMARY
# ============================================================

print("\nTEST 2: Executive Summary")

summary = build_executive_summary(
    dataset_profile={"rows": 200, "columns": 8},
    target_info={"problem_type": "regression"},
    model_results={"best_model": "Random Forest"},
    diagnosis_report={"status": "issues_found"},
)

assert summary["dataset"]["rows"] == 200
assert summary["problem_type"] == "regression"
assert summary["best_model"]["name"] == "Random Forest"

print("Executive summary passed.")


# ============================================================
# TEST 3: MISSING SECTIONS
# ============================================================

print("\nTEST 3: Missing Sections")

report = generate_report()

assert report["dataset_profile"] is None
assert report["evaluation"] is None
assert report["diagnosis"] is None

print("Missing sections handled correctly.")


# ============================================================
# TEST 4: JSON SERIALIZATION
# ============================================================

print("\nTEST 4: JSON Serialization")

report = generate_report(
    dataset_profile={"rows": 100},
    model_results={"score": 0.95},
)

json_text = json.dumps(report)
parsed = json.loads(json_text)

assert parsed["dataset_profile"]["rows"] == 100

print("JSON serialization passed.")


# ============================================================
# TEST 5: JSON EXPORT
# ============================================================

print("\nTEST 5: JSON Export")

with tempfile.TemporaryDirectory() as temp_dir:

    path = Path(temp_dir) / "report.json"

    result_path = export_json_report(
        report,
        str(path),
    )

    assert Path(result_path).exists()

    with open(result_path, encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded["report_version"] == "1.0"

print("JSON export passed.")


# ============================================================
# TEST 6: HTML GENERATION
# ============================================================

print("\nTEST 6: HTML Generation")

html_content = build_html_report(report)

assert "<html" in html_content
assert "ML Model Doctor" in html_content
assert "Dataset Profile" in html_content

print("HTML generation passed.")


# ============================================================
# TEST 7: HTML EXPORT
# ============================================================

print("\nTEST 7: HTML Export")

with tempfile.TemporaryDirectory() as temp_dir:

    path = Path(temp_dir) / "report.html"

    result_path = export_html_report(
        report,
        str(path),
    )

    assert Path(result_path).exists()

print("HTML export passed.")


# ============================================================
# TEST 8: DASHBOARD SUMMARY
# ============================================================

print("\nTEST 8: Dashboard Summary")

report = generate_report(
    dataset_profile={"rows": 150, "columns": 6},
    target_info={"problem_type": "classification"},
    model_results={"best_model": "Logistic Regression"},
)

dashboard = build_dashboard_summary(report)

assert dashboard["dataset"]["rows"] == 150
assert dashboard["problem_type"] == "classification"

print("Dashboard summary passed.")


# ============================================================
# TEST 9: COMPLETE REPORT SAVE
# ============================================================

print("\nTEST 9: Complete Report Save")

with tempfile.TemporaryDirectory() as temp_dir:

    paths = save_report(
        report,
        json_path=str(Path(temp_dir) / "report.json"),
        html_path=str(Path(temp_dir) / "report.html"),
    )

    assert Path(paths["json_path"]).exists()
    assert Path(paths["html_path"]).exists()

print("Complete report save passed.")


# ============================================================
# TEST 10: INVALID INPUT
# ============================================================

print("\nTEST 10: Invalid Input")

try:
    generate_report(dataset_profile="invalid")

    raise AssertionError("Expected ValueError")

except ValueError:
    print("Invalid input handled correctly.")


print("\n========== ALL REPORT GENERATOR TESTS COMPLETED ==========")


import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from services.analysis_service import analyze_dataset


# ============================================================
# TEST DATA
# ============================================================

def create_classification_dataset():
    """Create a synthetic binary classification dataset."""

    return pd.DataFrame({
        "age": [
            22, 25, 28, 31, 24, 27, 29, 32, 26, 30,
            23, 35, 33, 21, 34, 28, 25, 31, 29, 26,
            24, 32, 27, 36, 30, 22, 34, 28, 33, 25,
            29, 31, 23, 35, 26, 30, 32, 24, 27, 34,
        ],
        "salary": [
            25000, 30000, 35000, 40000, 28000,
            33000, 38000, 42000, 31000, 39000,
            27000, 45000, 43000, 24000, 46000,
            36000, 29000, 41000, 37000, 32000,
            26000, 44000, 34000, 48000, 39000,
            25000, 47000, 35000, 45000, 30000,
            38000, 42000, 27000, 49000, 31000,
            40000, 43000, 28000, 33000, 46000,
        ],
        "experience": [
            1, 2, 3, 5, 1, 3, 4, 6, 2, 5,
            1, 7, 6, 1, 7, 4, 2, 5, 4, 3,
            1, 6, 3, 8, 5, 1, 7, 3, 6, 2,
            4, 5, 1, 8, 2, 5, 6, 1, 3, 7,
        ],
        "hired": [
            "No", "No", "Yes", "Yes", "No",
            "Yes", "Yes", "Yes", "No", "Yes",
            "No", "Yes", "Yes", "No", "Yes",
            "Yes", "No", "Yes", "Yes", "Yes",
            "No", "Yes", "Yes", "Yes", "Yes",
            "No", "Yes", "Yes", "Yes", "No",
            "Yes", "Yes", "No", "Yes", "No",
            "Yes", "Yes", "No", "Yes", "Yes",
        ],
    })


def create_regression_dataset():
    """Create a synthetic regression dataset."""

    return pd.DataFrame({
        "area": list(range(500, 2500, 50)),
        "bedrooms": [
            1 + (i % 5) for i in range(40)
        ],
        "age": [
            2 + (i % 20) for i in range(40)
        ],
        "price": [
            100000 + (i * 3500) + ((i % 4) * 2000)
            for i in range(40)
        ],
    })


# ============================================================
# TEST 1: CLASSIFICATION WORKFLOW
# ============================================================

def test_classification_workflow():

    print("\nTEST 1: Classification Workflow")

    df = create_classification_dataset()

    result = analyze_dataset(
        df,
        target_column="hired",
        max_models=2,
        generate_reports=False,
    )

    assert result["status"] == "success"
    assert result["problem"]["type"] == "binary_classification"
    assert result["problem"]["target"] == "hired"
    assert len(result["models"]) > 0
    assert result["recommendation"]["recommended_model"]

    print("Classification workflow passed.")


# ============================================================
# TEST 2: REGRESSION WORKFLOW
# ============================================================

def test_regression_workflow():

    print("\nTEST 2: Regression Workflow")

    df = create_regression_dataset()

    result = analyze_dataset(
        df,
        target_column="price",
        user_choice="regression",
        max_models=2,
        generate_reports=False,
    )

    assert result["status"] == "success"
    assert result["problem"]["type"] == "regression"
    assert result["problem"]["target"] == "price"
    assert len(result["models"]) > 0

    print("Regression workflow passed.")


# ============================================================
# TEST 3: AMBIGUOUS TARGET
# ============================================================

def test_ambiguous_target():

    print("\nTEST 3: Ambiguous Target Confirmation")

    df = pd.DataFrame({
        "experience": list(range(1, 31)),
        "study_hours": list(range(2, 32)),
        "rating": [1, 2, 3, 4, 5] * 6,
    })

    result = analyze_dataset(
        df,
        target_column="rating",
        generate_reports=False,
    )

    assert result["status"] == "target_confirmation_required"
    assert result["target_info"]["requires_confirmation"] is True

    print("Ambiguous target handled correctly.")


# ============================================================
# TEST 4: INVALID TARGET COLUMN
# ============================================================

def test_invalid_target():

    print("\nTEST 4: Invalid Target Column")

    df = create_classification_dataset()

    try:
        analyze_dataset(
            df,
            target_column="unknown_target",
            generate_reports=False,
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert "does not exist" in str(exc)

    print("Invalid target handled correctly.")


# ============================================================
# TEST 5: MISSING TARGET VALUES
# ============================================================


def test_missing_target_values():

    print("\nTEST 5: Missing Target Values")

    df = create_classification_dataset()

    original_rows = len(df)

    df.loc[0, "hired"] = None
    df.loc[1, "hired"] = None

    expected_rows = (
        df.dropna(subset=["hired"])
        .drop_duplicates()
        .shape[0]
    )

    result = analyze_dataset(
        df,
        target_column="hired",
        max_models=2,
        generate_reports=False,
    )

    assert result["status"] == "success"

    assert result["profile"]["target_missing_values"] == 2

    assert result["cleaning_report"]["target_rows_removed"] == 2

    assert result["dataset"]["rows"] == expected_rows

    assert result["dataset"]["rows"] <= original_rows - 2

    print("Original rows:", original_rows)
    print("Expected usable rows:", expected_rows)
    print("Actual model rows:", result["dataset"]["rows"])

    print("Missing target values handled correctly.")

# ============================================================
# TEST 6: REPORT GENERATION
# ============================================================

def test_report_generation():

    print("\nTEST 6: Report Generation")

    df = create_classification_dataset()

    temp_dir = tempfile.mkdtemp(
        prefix="ml_doctor_test_"
    )

    try:

        result = analyze_dataset(
            df,
            target_column="hired",
            max_models=2,
            generate_reports=True,
            report_directory=temp_dir,
        )

        assert result["status"] == "success"

        json_path = result["report_paths"]["json"]
        html_path = result["report_paths"]["html"]

        assert os.path.isfile(json_path)
        assert os.path.isfile(html_path)

        with open(json_path, "r", encoding="utf-8") as file:
            report_data = json.load(file)

        assert report_data["executive_summary"]

        print("JSON and HTML report generation passed.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# TEST 7: INTERNAL TRAINED MODELS
# ============================================================

def test_trained_models():

    print("\nTEST 7: Trained Models Availability")

    df = create_classification_dataset()

    result = analyze_dataset(
        df,
        target_column="hired",
        max_models=2,
        generate_reports=False,
    )

    trained_models = result["_trained_models"]

    assert isinstance(trained_models, dict)
    assert len(trained_models) > 0

    for model in trained_models.values():
        assert hasattr(model, "predict")

    print("Trained models are available.")


# ============================================================
# TEST 8: PUBLIC RESULT JSON SERIALIZATION
# ============================================================

def test_public_result_serialization():

    print("\nTEST 8: Public Result Serialization")

    df = create_classification_dataset()

    result = analyze_dataset(
        df,
        target_column="hired",
        max_models=2,
        generate_reports=False,
    )

    public_result = {
        key: value
        for key, value in result.items()
        if key not in (
            "cleaned_dataframe",
            "_trained_models",
            "_baseline_pipeline",
            "report",
        )
    }

    # The report is intentionally omitted here because
    # its own JSON export is tested separately.

    serialized = json.dumps(
        public_result,
        allow_nan=False,
    )

    assert isinstance(serialized, str)

    print("Public result serialization passed.")


# ============================================================
# TEST 9: EMPTY DATASET
# ============================================================

def test_empty_dataset():

    print("\nTEST 9: Empty Dataset")

    df = pd.DataFrame()

    try:
        analyze_dataset(
            df,
            target_column="target",
            generate_reports=False,
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError as exc:
        assert "empty" in str(exc).lower()

    print("Empty dataset handled correctly.")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    test_classification_workflow()
    test_regression_workflow()
    test_ambiguous_target()
    test_invalid_target()
    test_missing_target_values()
    test_report_generation()
    test_trained_models()
    test_public_result_serialization()
    test_empty_dataset()

    print(
        "\n========== "
        "ALL ANALYSIS SERVICE TESTS COMPLETED "
        "=========="
    )

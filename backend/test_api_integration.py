
import io
import json
import pandas as pd
from ml.target_detector import detect_target_type
from app import app


def run_api_tests():

    client = app.test_client()

    print("\nTEST 1: API Health")

    response = client.get("/api/test")

    assert response.status_code == 200
    assert response.json["status"] == "success"

    print("API health passed.")

    # Create a synthetic classification dataset.
    # This is only for testing the application workflow.
    df = pd.DataFrame({
        "age": list(range(20, 100)),
        "experience": list(range(80)),
        "salary": [25000 + i * 1500 for i in range(80)],
        "score": [i * 2 + (i % 5) for i in range(80)],
        "hired": [1 if i % 2 == 0 else 0 for i in range(80)]
    })

    print("\nTEST 2: Upload CSV")

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = client.post(
        "/api/upload",
        data={
            "file": (
                csv_buffer,
                "integration_test.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200, response.json
    assert response.json["status"] == "success"

    filename = response.json["file"]["filename"]

    print("Upload passed:", filename)
    
    print("\nDEBUG: Target Detection")

    print("Target values:")
    print(df["hired"].value_counts())

    print("Target dtype:", df["hired"].dtype)
    print("Unique values:", df["hired"].nunique())

    target_result = detect_target_type(
        df["hired"],
        target_name="hired"
    )

    print("Target detector result:")
    print(target_result)

    print("\nTEST 3: Analyze Dataset")

    response = client.post(
        "/api/analyze",
        json={
            "filename": filename,
            "target_column": "hired",
            "problem_type": "binary_classification",
            "max_models": 3,
            "test_size": 0.20
        }
    )

    assert response.status_code == 200, response.json

    result = response.json

    assert result["status"] == "success"
    assert "models" in result
    assert "evaluation" in result
    assert "recommendation" in result
    assert "diagnosis" in result

    print("Analysis passed.")
    
    print("\n========== FRONTEND RESPONSE STRUCTURE ==========")

    print("Problem:")
    print(json.dumps(result.get("problem"), indent=2))

    print("\nEvaluation:")
    print(json.dumps(result.get("evaluation"), indent=2))

    print("\nFirst model:")
    print(json.dumps(
        result.get("models", [{}])[0],
        indent=2
    ))

    print("\nModel comparison keys:")
    print(list(result.get("model_comparison", {}).keys()))

    print("\nRecommendation:")
    print(json.dumps(result.get("recommendation"), indent=2))

    print("\nDiagnosis:")
    print(json.dumps(result.get("diagnosis"), indent=2))

    print("\nProfile keys:")
    print(list(result.get("profile", {}).keys()))

    print("\nTEST 4: Evaluation Report")

    evaluation = result["evaluation"]

    assert "details" in evaluation

    details = evaluation["details"]

    assert details.get("available") is True
    assert "generalization" in details
    assert "baseline_comparison" in details
    assert "underfitting" in details
    assert "warnings" in details

    print("Evaluation integration passed.")

    print("\nTEST 5: Recommendation")

    recommendation = result["recommendation"]

    assert recommendation["recommended_model"] is not None
    assert recommendation["cv_score"] is not None
    assert recommendation["test_score"] is not None

    print("Recommendation passed.")

    print("\nTEST 6: Public Serialization")

    serialized = json.dumps(result)

    assert "_trained_models" not in serialized
    assert "_baseline_pipeline" not in serialized

    print("Public serialization passed.")

    print("\nTEST 7: Download Cleaned Dataset")

    cleaned_file = result["cleaned_file"]

    response = client.get(
        f"/api/download-cleaned/{cleaned_file}"
    )

    assert response.status_code == 200

    print("Cleaned CSV download passed.")

    print("\nTEST 8: Download JSON Report")

    json_report = result["reports"]["json"]

    assert json_report is not None

    response = client.get(
        f"/api/download-report/json/{json_report}"
    )

    assert response.status_code == 200

    json_content = json.loads(response.data.decode("utf-8"))

    print("JSON report download passed.")

    print("\nTEST 9: Download HTML Report")

    html_report = result["reports"]["html"]

    assert html_report is not None

    response = client.get(
        f"/api/download-report/html/{html_report}"
    )

    assert response.status_code == 200
    assert len(response.data) > 0

    print("HTML report download passed.")

    print("\n========== ALL API INTEGRATION TESTS PASSED ==========")


if __name__ == "__main__":
    run_api_tests()

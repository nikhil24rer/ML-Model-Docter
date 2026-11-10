
from ml.recommendations import generate_recommendation


def run_tests():

    print("\nTEST 1: Binary Classification")

    comparison = {
        "best_model": "Random Forest",
        "problem_type": "binary_classification",
        "primary_metric": "macro F1",
        "selection_metric": "f1_macro",
        "best_score": 0.91,
        "best_test_score": 0.90,
        "baseline": {"f1": 0.50},
        "dataset_info": {"is_imbalanced": False},
        "models": [
            {"model_id": "rf", "status": "success", "score": 0.90}
        ]
    }

    profile = {
        "original_missing_values": 5,
        "original_duplicate_rows": 2,
        "target_missing_values": 1,
        "target_distribution": {"0": 50, "1": 50}
    }

    result = generate_recommendation(comparison, profile)

    assert result["recommended_model"] == "Random Forest"
    assert result["cv_score"] == 0.91
    assert result["test_score"] == 0.90

    print("Binary classification passed.")

    print("\nTEST 2: Multiclass Classification")

    comparison["problem_type"] = "multiclass_classification"

    result = generate_recommendation(comparison, profile)

    assert result["recommended_model"] == "Random Forest"

    print("Multiclass classification passed.")

    print("\nTEST 3: Regression")

    regression_comparison = {
        "best_model": "Linear Regression",
        "problem_type": "regression",
        "primary_metric": "R2",
        "best_score": 0.95,
        "best_test_score": 0.92,
        "baseline": {"r2": 0.20},
        "models": [
            {"model_id": "lr", "status": "success", "score": 0.92}
        ]
    }

    result = generate_recommendation(
        regression_comparison, {}
    )

    assert result["recommended_model"] == "Linear Regression"
    assert result["test_score"] == 0.92

    print("Regression passed.")

    print("\nTEST 4: No Successful Models")

    empty_comparison = {
        "best_model": None,
        "problem_type": "regression",
        "models": []
    }

    result = generate_recommendation(empty_comparison, {})

    assert result["recommended_model"] is None

    print("No successful models handled correctly.")

    print("\nALL RECOMMENDATION TESTS PASSED!")


if __name__ == "__main__":
    run_tests()

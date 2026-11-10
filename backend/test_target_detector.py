
import pandas as pd

from ml.target_detector import detect_target_type


# ==========================================
# TEST 1: BINARY CLASSIFICATION
# ==========================================

df1 = pd.Series(
    ["Yes", "No", "Yes", "No", "Yes", "No"]
)

result1 = detect_target_type(
    df1,
    target_name="Placement"
)

print("\nTEST 1: Binary Classification")
print("Detected:", result1["problem_type"])
print("Reason:", result1["reason"])


# ==========================================
# TEST 2: MULTICLASS CLASSIFICATION
# ==========================================

df2 = pd.Series(
    ["Cat", "Dog", "Bird", "Cat", "Dog", "Bird"]
)

result2 = detect_target_type(
    df2,
    target_name="Animal"
)

print("\nTEST 2: Multiclass Classification")
print("Detected:", result2["problem_type"])
print("Classes:", result2["class_analysis"]["class_count"])


# ==========================================
# TEST 3: REGRESSION
# ==========================================

df3 = pd.Series(
    [float(x) + 0.5 for x in range(100)]
)

result3 = detect_target_type(
    df3,
    target_name="House_Price"
)

print("\nTEST 3: Regression")
print("Detected:", result3["problem_type"])
print("Statistics:", result3["numeric_analysis"])


# ==========================================
# TEST 4: AMBIGUOUS INTEGER TARGET
# ==========================================

df4 = pd.Series(
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
)

result4 = detect_target_type(
    df4,
    target_name="Rating"
)

print("\nTEST 4: Ambiguous Target")
print("Detected:", result4["problem_type"])
print("Confirmation Required:", result4["requires_confirmation"])
print("Reason:", result4["reason"])


# ==========================================
# TEST 5: IMBALANCED CLASSIFICATION
# ==========================================

df5 = pd.Series(
    ["No"] * 90 + ["Yes"] * 10
)

result5 = detect_target_type(
    df5,
    target_name="Churn"
)

print("\nTEST 5: Imbalanced Classification")
print("Detected:", result5["problem_type"])
print("Distribution:", result5["class_analysis"])
print("Imbalanced:", result5["class_analysis"]["is_imbalanced"])


# ==========================================
# TEST 6: NUMERICAL BINARY TARGET
# ==========================================

df6 = pd.Series(
    [0, 1, 0, 1, 0, 1]
)

result6 = detect_target_type(
    df6,
    target_name="Outcome"
)

print("\nTEST 6: Numerical Binary Target")
print("Detected:", result6["problem_type"])
print("Confirmation Required:", result6["requires_confirmation"])


# ==========================================
# TEST 7: USER CONFIRMS REGRESSION
# ==========================================

result7 = detect_target_type(
    df4,
    target_name="Rating",
    user_choice="regression"
)

print("\nTEST 7: User Confirmation")
print("Detected:", result7["problem_type"])
print("Confidence:", result7["confidence"])


# ==========================================
# TEST 8: INVALID SINGLE CLASS TARGET
# ==========================================

df8 = pd.Series(
    ["Yes", "Yes", "Yes", "Yes"]
)

result8 = detect_target_type(
    df8,
    target_name="Outcome"
)

print("\nTEST 8: Invalid Target")
print("Detected:", result8["problem_type"])
print("Reason:", result8["reason"])


# ==========================================
# FINAL TEST SUMMARY
# ==========================================

results = [
    result1,
    result2,
    result3,
    result4,
    result5,
    result6,
    result7,
    result8
]

print("\n========== TEST SUMMARY ==========")

for index, result in enumerate(results, start=1):

    print(
        f"Test {index}: "
        f"{result['problem_type']} "
        f"| Confirmation: "
        f"{result['requires_confirmation']}"
    )

print("\nAll test cases executed successfully.")

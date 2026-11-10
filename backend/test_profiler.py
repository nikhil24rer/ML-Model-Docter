
import pandas as pd

from ml.data_profiler import profile_dataset


# ==========================================
# CREATE SAMPLE DATASET
# ==========================================

df = pd.DataFrame({

    "Age": [21, 22, 24, 25, 65, None],

    "Salary": [
        25000,
        30000,
        35000,
        40000,
        500000,
        45000
    ],

    "City": [
        "Ludhiana",
        "Delhi",
        "Mohali",
        "Chandigarh",
        "Delhi",
        None
    ],

    "Customer_ID": [
        101, 102, 103, 104, 105, 106
    ],

    "Active": [
        True, True, False, True, False, True
    ],

    "Constant_Column": [
        1, 1, 1, 1, 1, 1
    ]

})


# ==========================================
# RUN DATASET PROFILING
# ==========================================

report = profile_dataset(df)


# ==========================================
# DISPLAY BASIC INFORMATION
# ==========================================

print("\n========== DATASET PROFILE ==========")

print("Rows:", report["rows"])

print("Columns:", report["columns"])

print("Total Cells:", report["total_cells"])


# ==========================================
# DISPLAY COLUMN TYPES
# ==========================================

print("\n========== COLUMN TYPES ==========")

print("Numeric:", report["numeric_columns"])

print("Categorical:", report["categorical_columns"])

print("Boolean:", report["boolean_columns"])

print("Datetime:", report["datetime_columns"])


# ==========================================
# DISPLAY DATA QUALITY
# ==========================================

print("\n========== DATA QUALITY ==========")

print("Missing Values:", report["missing_values"])

print("Missing Percentage:", report["missing_percentage"])

print("Duplicate Rows:", report["duplicate_rows"])

print("Constant Columns:", report["constant_columns"])

print("Near Constant:", report["near_constant_columns"])


# ==========================================
# DISPLAY FEATURE ANALYSIS
# ==========================================

print("\n========== FEATURE ANALYSIS ==========")

print(
    "Identifier Candidates:",
    report["identifier_candidates"]
)

print(
    "High Cardinality:",
    report["high_cardinality_columns"]
)

print(
    "Outlier Values:",
    report["outlier_values"]
)


# ==========================================
# DISPLAY NUMERICAL STATISTICS
# ==========================================

print("\n========== NUMERICAL SUMMARY ==========")

for column, statistics in report["numerical_summary"].items():

    print("\nColumn:", column)

    for metric, value in statistics.items():

        print(metric, ":", value)


# ==========================================
# DISPLAY WARNINGS
# ==========================================

print("\n========== WARNINGS ==========")

for warning in report["warnings"]:

    print(
        warning["type"],
        "|",
        warning["severity"],
        "|",
        warning["message"]
    )


# ==========================================
# FINAL RESULT
# ==========================================

print("\n========== TEST COMPLETED ==========")

print("Dataset profiling executed successfully.")

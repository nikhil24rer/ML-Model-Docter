import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "../../dataset/student_placement.csv"

df = pd.read_csv(file_path)

print("========== DATA QUALITY REPORT ==========")


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n----- DATASET INFORMATION -----")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(list(df.columns))


# ==========================================
# 3. DATA TYPES
# ==========================================

print("\n----- DATA TYPES -----")

print(df.dtypes)


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n----- MISSING VALUES -----")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

if total_missing > 0:
    print("\n⚠️ Missing values detected!")
else:
    print("\n✅ No missing values detected.")


# ==========================================
# 5. DUPLICATE ROWS
# ==========================================

print("\n----- DUPLICATE ROWS -----")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates > 0:
    print("⚠️ Duplicate rows detected!")
else:
    print("✅ No duplicate rows detected.")


# ==========================================
# 6. NUMERICAL COLUMNS
# ==========================================

print("\n----- NUMERICAL COLUMNS -----")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print(list(numeric_columns))


# ==========================================
# 7. OUTLIER DETECTION USING IQR
# ==========================================

print("\n----- OUTLIER DETECTION -----")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(
        f"{column}: {len(outliers)} outlier(s)"
    )


# ==========================================
# 8. TARGET CLASS DISTRIBUTION
# ==========================================

print("\n----- TARGET DISTRIBUTION -----")

if "Placement" in df.columns:

    class_distribution = df["Placement"].value_counts()

    print(class_distribution)

    class_ratio = (
        class_distribution.min()
        / class_distribution.max()
    )

    if class_ratio < 0.5:
        print("\n⚠️ Possible class imbalance detected!")
    else:
        print("\n✅ No major class imbalance detected.")


# ==========================================
# 9. SUMMARY
# ==========================================

print("\n========== SUMMARY ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Missing values:", total_missing)
print("Duplicate rows:", duplicates)

print("\n========== DATA QUALITY CHECK COMPLETE ==========")
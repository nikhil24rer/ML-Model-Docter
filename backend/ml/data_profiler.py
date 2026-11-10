import re
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NEAR_CONSTANT_THRESHOLD = 0.98
HIGH_CARDINALITY_THRESHOLD = 50
HIGH_CARDINALITY_RATIO = 0.50
OUTLIER_IQR_MULTIPLIER = 1.5
DATETIME_PARSE_THRESHOLD = 0.90


# --------------------------------------------------
# HELPER: SAFE VALUE CONVERSION
# --------------------------------------------------

def safe_value(value):
    """
    Convert NumPy and Pandas values into JSON-safe values.
    """

    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (int, float, str, bool)):
        return value

    return str(value)


# --------------------------------------------------
# HELPER: DETECT DATETIME COLUMNS
# --------------------------------------------------

def is_datetime_candidate(series):
    """
    Identify strings that appear to contain dates.

    Existing datetime columns are handled separately.
    """

    if is_datetime64_any_dtype(series):
        return True

    if not (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        return False

    sample = series.dropna()

    if sample.empty:
        return False

    sample = sample.astype(str).head(1000)

    date_pattern = re.compile(
        r"[-/:T]|"
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
        re.IGNORECASE
    )

    matching_values = sample.apply(
        lambda value: bool(date_pattern.search(value))
    )

    if matching_values.mean() < 0.5:
        return False

    try:
        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed"
        )

        success_rate = parsed.notna().mean()

        return bool(
            success_rate >= DATETIME_PARSE_THRESHOLD
        )

    except (ValueError, TypeError, OverflowError):
        return False


# --------------------------------------------------
# HELPER: IDENTIFIER DETECTION
# --------------------------------------------------

def detect_identifier(series, column_name):
    """
    Identify potential ID columns using their names
    and uniqueness ratio.

    This is only a warning, not automatic deletion.
    """

    name = str(column_name).strip().lower()

    id_patterns = [
        r"^id$",
        r"_id$",
        r"^id_",
        r"uuid",
        r"identifier",
        r"customer_number",
        r"employee_number",
        r"record_number",
    ]

    name_match = any(
        re.search(pattern, name)
        for pattern in id_patterns
    )

    total = len(series)
    unique_count = series.nunique(dropna=True)

    unique_ratio = (
        unique_count / total
        if total > 0 else 0
    )

    high_uniqueness = (
        total >= 10
        and unique_ratio >= 0.95
    )

    return bool(name_match or high_uniqueness)


# --------------------------------------------------
# HELPER: COLUMN TYPE DETECTION
# --------------------------------------------------

def detect_column_type(series):
    """
    Identify the general type of a column.
    """

    if is_bool_dtype(series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numeric"

    if is_datetime_candidate(series):
        return "datetime_candidate"

    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        return "categorical"

    return "other"


# --------------------------------------------------
# HELPER: OUTLIER ANALYSIS
# --------------------------------------------------

def analyze_outliers(series):
    """
    Detect numerical outliers using the IQR method.

    Outliers are flagged, never deleted here.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(values) < 4:
        return {
            "count": 0,
            "percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None
        }

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr) or iqr <= 0:
        return {
            "count": 0,
            "percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None
        }

    lower_bound = (
        q1 - OUTLIER_IQR_MULTIPLIER * iqr
    )

    upper_bound = (
        q3 + OUTLIER_IQR_MULTIPLIER * iqr
    )

    mask = (
        (values < lower_bound)
        | (values > upper_bound)
    )

    count = int(mask.sum())

    percentage = (
        count / len(values) * 100
        if len(values) else 0
    )

    return {
        "count": count,
        "percentage": round(float(percentage), 2),
        "lower_bound": safe_value(lower_bound),
        "upper_bound": safe_value(upper_bound)
    }


# --------------------------------------------------
# HELPER: NUMERICAL STATISTICS
# --------------------------------------------------

def numerical_statistics(series):
    """
    Calculate useful statistics for numerical features.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if values.empty:
        return {}

    return {
        "mean": safe_value(values.mean()),
        "median": safe_value(values.median()),
        "std": safe_value(values.std()),
        "min": safe_value(values.min()),
        "max": safe_value(values.max()),
        "skewness": safe_value(values.skew()),
        "variance": safe_value(values.var())
    }


# --------------------------------------------------
# MAIN FUNCTION: PROFILE DATASET
# --------------------------------------------------

def profile_dataset(df):
    """
    Generate a comprehensive, dataset-driven
    data-quality and statistical report.

    This function does not modify the original dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    if df.shape[1] == 0:
        raise ValueError(
            "The dataset does not contain columns."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names detected. "
            "Please make column names unique before analysis."
        )

    rows, columns = df.shape

    # ----------------------------------------------
    # OVERALL DATASET STATISTICS
    # ----------------------------------------------

    total_cells = rows * columns

    missing_per_column = df.isna().sum()

    missing_values = int(
        missing_per_column.sum()
    )

    missing_percentage = (
        missing_values / total_cells * 100
        if total_cells else 0
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # ----------------------------------------------
    # COLUMN TYPE DETECTION
    # ----------------------------------------------

    numeric_columns = []
    categorical_columns = []
    boolean_columns = []
    datetime_columns = []
    datetime_candidates = []
    other_columns = []

    column_types = {}

    for column in df.columns:

        series = df[column]

        column_type = detect_column_type(series)

        column_types[column] = column_type

        if column_type == "numeric":
            numeric_columns.append(column)

        elif column_type == "categorical":
            categorical_columns.append(column)

        elif column_type == "boolean":
            boolean_columns.append(column)

        elif column_type == "datetime":
            datetime_columns.append(column)

        elif column_type == "datetime_candidate":
            datetime_candidates.append(column)

        else:
            other_columns.append(column)

    # ----------------------------------------------
    # PER-COLUMN ANALYSIS
    # ----------------------------------------------

    column_info = []

    constant_columns = []
    near_constant_columns = []
    identifier_columns = []
    high_cardinality_columns = []

    outlier_columns = {}
    numerical_summary = {}

    for column in df.columns:

        series = df[column]

        name = str(column)

        missing_count = int(
            series.isna().sum()
        )

        unique_count = int(
            series.nunique(dropna=True)
        )

        unique_ratio = (
            unique_count / rows
            if rows else 0
        )

        is_constant = (
            series.nunique(dropna=False) <= 1
        )

        # Near-constant detection
        frequencies = series.value_counts(
            normalize=True,
            dropna=False
        )

        dominant_ratio = (
            float(frequencies.iloc[0])
            if not frequencies.empty else 0
        )

        is_near_constant = (
            not is_constant
            and dominant_ratio >= NEAR_CONSTANT_THRESHOLD
        )

        # Identifier detection
        is_identifier = detect_identifier(
            series,
            column
        )

        # High-cardinality detection
        is_high_cardinality = (
            column_types[column] == "categorical"
            and unique_count >= HIGH_CARDINALITY_THRESHOLD
            and unique_ratio >= HIGH_CARDINALITY_RATIO
        )

        # Outlier analysis
        outlier_info = {
            "count": 0,
            "percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None
        }

        if column_types[column] == "numeric":

            outlier_info = analyze_outliers(series)

            outlier_columns[name] = (
                outlier_info["count"]
            )

            numerical_summary[name] = (
                numerical_statistics(series)
            )

        # Save detected columns
        if is_constant:
            constant_columns.append(name)

        if is_near_constant:
            near_constant_columns.append(name)

        if is_identifier:
            identifier_columns.append(name)

        if is_high_cardinality:
            high_cardinality_columns.append(name)

        # Column report
        column_info.append({

            "name": name,

            "dtype": str(series.dtype),

            "detected_type": column_types[column],

            "missing": missing_count,

            "missing_percentage": round(
                missing_count / rows * 100,
                2
            ),

            "unique_values": unique_count,

            "unique_ratio": round(
                unique_ratio,
                4
            ),

            "is_numeric": (
                column_types[column] == "numeric"
            ),

            "is_constant": bool(is_constant),

            "is_near_constant": bool(
                is_near_constant
            ),

            "is_identifier_candidate": bool(
                is_identifier
            ),

            "is_high_cardinality": bool(
                is_high_cardinality
            ),

            "outliers": outlier_info["count"],

            "outlier_percentage": (
                outlier_info["percentage"]
            ),

            "outlier_bounds": {
                "lower": outlier_info["lower_bound"],
                "upper": outlier_info["upper_bound"]
            },

            "dominant_value_percentage": round(
                dominant_ratio * 100,
                2
            )

        })

    # ----------------------------------------------
    # DUPLICATE COLUMN DETECTION
    # ----------------------------------------------

    duplicate_columns = []

    column_names = list(df.columns)

    for i in range(len(column_names)):

        for j in range(i + 1, len(column_names)):

            first_column = column_names[i]
            second_column = column_names[j]

            if df[first_column].equals(
                df[second_column]
            ):
                duplicate_columns.append({
                    "column": str(second_column),
                    "duplicate_of": str(first_column)
                })

    # ----------------------------------------------
    # OUTLIER SUMMARY
    # ----------------------------------------------

    total_outlier_values = int(
        sum(outlier_columns.values())
    )

    outlier_row_indices = set()

    for column in numeric_columns:

        info = analyze_outliers(df[column])

        if info["count"] == 0:
            continue

        lower = info["lower_bound"]
        upper = info["upper_bound"]

        if lower is None or upper is None:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        mask = (
            (values < lower)
            | (values > upper)
        ).fillna(False)

        outlier_row_indices.update(
            df.index[mask].tolist()
        )

    outlier_rows = len(outlier_row_indices)

    outlier_details = [
        {
            "column": column,
            "count": count
        }

        for column, count in outlier_columns.items()

        if count > 0
    ]

    outlier_details.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    # ----------------------------------------------
    # DATASET QUALITY WARNINGS
    # ----------------------------------------------

    warnings = []

    if missing_values > 0:

        warnings.append({
            "type": "missing_values",
            "severity": "high" if missing_percentage >= 30 else "medium",
            "message": (
                f"{missing_values} missing values detected "
                f"({missing_percentage:.2f}% of all cells)."
            )
        })

    if duplicate_rows > 0:

        warnings.append({
            "type": "duplicate_rows",
            "severity": "medium",
            "message": (
                f"{duplicate_rows} duplicate rows detected. "
                "Review before removing them."
            )
        })

    if constant_columns:

        warnings.append({
            "type": "constant_columns",
            "severity": "medium",
            "message": (
                f"{len(constant_columns)} constant columns detected."
            )
        })

    if near_constant_columns:

        warnings.append({
            "type": "near_constant_columns",
            "severity": "low",
            "message": (
                f"{len(near_constant_columns)} near-constant "
                "columns detected."
            )
        })

    if identifier_columns:

        warnings.append({
            "type": "identifier_candidates",
            "severity": "low",
            "message": (
                "Potential identifier columns detected. "
                "Review whether they should be used as features."
            )
        })

    if high_cardinality_columns:

        warnings.append({
            "type": "high_cardinality",
            "severity": "medium",
            "message": (
                "High-cardinality categorical columns detected."
            )
        })

    if datetime_candidates:

        warnings.append({
            "type": "datetime_candidates",
            "severity": "low",
            "message": (
                "Some string columns appear to contain dates. "
                "Review their data types."
            )
        })

    if total_outlier_values > 0:

        warnings.append({
            "type": "outliers",
            "severity": "medium",
            "message": (
                f"{total_outlier_values} potential numerical "
                "outlier values detected using IQR."
            )
        })

    # ----------------------------------------------
    # FINAL DATASET PROFILE
    # ----------------------------------------------

    return {

        # Dataset information
        "rows": int(rows),

        "columns": int(columns),

        "total_cells": int(total_cells),

        "column_names": [
            str(column) for column in df.columns
        ],

        # Data types
        "numeric_columns": [
            str(column) for column in numeric_columns
        ],

        "categorical_columns": [
            str(column) for column in categorical_columns
        ],

        "boolean_columns": [
            str(column) for column in boolean_columns
        ],

        "datetime_columns": [
            str(column) for column in datetime_columns
        ],

        "datetime_candidates": [
            str(column) for column in datetime_candidates
        ],

        "other_columns": [
            str(column) for column in other_columns
        ],

        "data_type_summary": {
            "numeric": len(numeric_columns),
            "categorical": len(categorical_columns),
            "boolean": len(boolean_columns),
            "datetime": len(datetime_columns),
            "datetime_candidates": len(datetime_candidates),
            "other": len(other_columns)
        },

        # Missing values
        "missing_values": missing_values,

        "missing_percentage": round(
            float(missing_percentage), 2
        ),

        "columns_with_missing": int(
            (missing_per_column > 0).sum()
        ),

        # Duplicate records
        "duplicate_rows": duplicate_rows,

        "duplicate_columns": duplicate_columns,

        # Outliers
        "outlier_values": total_outlier_values,

        "outlier_rows": int(outlier_rows),

        "outlier_details": outlier_details,

        # Feature quality
        "constant_columns": constant_columns,

        "near_constant_columns": near_constant_columns,

        "identifier_candidates": identifier_columns,

        "high_cardinality_columns": high_cardinality_columns,

        # Statistics
        "numerical_summary": numerical_summary,

        # Individual column information
        "column_info": column_info,

        # Warnings
        "warnings": warnings

    }
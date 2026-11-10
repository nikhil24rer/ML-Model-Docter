
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
)
from pandas.api.types import is_datetime64_any_dtype

# ==================================================
# CONFIGURATION
# ==================================================

MIN_RECORDS = 2

MIN_CLASSES = 2

MAX_CLASSES_FOR_AMBIGUOUS_INTEGER = 10

HIGH_CARDINALITY_THRESHOLD = 50

IMBALANCE_RATIO_THRESHOLD = 5.0

NUMERIC_STRING_THRESHOLD = 0.95


# ==================================================
# HELPER 1: SAFE VALUE CONVERSION
# ==================================================

def safe_value(value):
    """
    Convert NumPy and Pandas values into
    JSON-compatible Python values.
    """

    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ==================================================
# HELPER 2: DETECT NUMERIC-LIKE STRING TARGET
# ==================================================

def is_numeric_string(series):
    """
    Check whether a categorical-looking target
    contains values that can be converted to numbers.

    Example:
        ['10', '20', '30']

    Returns True when enough non-missing values
    can be converted successfully.
    """

    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False

    values = series.dropna()

    if values.empty:
        return False

    converted = pd.to_numeric(
        values.astype(str).str.strip(),
        errors="coerce"
    )

    success_rate = converted.notna().mean()

    return bool(
        success_rate >= NUMERIC_STRING_THRESHOLD
    )


# ==================================================
# HELPER 3: TARGET DISTRIBUTION
# ==================================================

def get_target_distribution(series):
    """
    Calculate the distribution of target classes.

    Useful for classification problems.
    """

    counts = series.value_counts(
        dropna=True
    )

    total = int(counts.sum())

    if total == 0:
        return {
            "class_count": 0,
            "class_distribution": [],
            "majority_class": None,
            "majority_percentage": 0.0,
            "minority_percentage": 0.0,
            "imbalance_ratio": None,
            "is_imbalanced": False
        }

    distribution = []

    for label, count in counts.items():

        percentage = (
            count / total * 100
        )

        distribution.append({

            "class": safe_value(label),

            "count": int(count),

            "percentage": round(
                float(percentage), 2
            )

        })

    majority_count = int(counts.iloc[0])

    minority_count = int(counts.iloc[-1])

    majority_percentage = (
        majority_count / total * 100
    )

    minority_percentage = (
        minority_count / total * 100
    )

    imbalance_ratio = (
        majority_count / minority_count
        if minority_count > 0 else None
    )

    is_imbalanced = (
        imbalance_ratio is not None
        and imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD
    )

    return {

        "class_count": int(len(counts)),

        "class_distribution": distribution,

        "majority_class": safe_value(counts.index[0]),

        "majority_percentage": round(
            float(majority_percentage), 2
        ),

        "minority_percentage": round(
            float(minority_percentage), 2
        ),

        "imbalance_ratio": (
            round(float(imbalance_ratio), 2)
            if imbalance_ratio is not None
            else None
        ),

        "is_imbalanced": bool(is_imbalanced)

    }


# ==================================================
# HELPER 4: NUMERICAL TARGET STATISTICS
# ==================================================

def get_numeric_statistics(series):
    """
    Calculate statistics for numerical targets.
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

        "mean": float(values.mean()),

        "median": float(values.median()),

        "std": float(values.std()) if len(values) > 1 else 0.0,

        "min": float(values.min()),

        "max": float(values.max()),

        "unique_values": int(values.nunique()),

        "skewness": (
            float(values.skew())
            if len(values) > 2
            else 0.0
        )

    }


# ==================================================
# HELPER 5: RESULT GENERATOR
# ==================================================

def create_result(
    problem_type,
    suggested_problem_type,
    confidence,
    requires_confirmation,
    reason,
    target_name,
    series,
    warnings=None
):
    """
    Generate a standardized target detection response.
    """

    clean_series = series.dropna()
    
    if is_datetime64_any_dtype(clean_series):

        return create_result(
            problem_type="unsupported",
            suggested_problem_type=None,
            confidence="none",
            requires_confirmation=False,
            reason=(
                "Datetime targets require a dedicated "
                "time-series forecasting workflow."
            ),
            target_name=target_name,
            series=series,
            warnings=[
                "Time-series forecasting is not yet supported."
            ]
        )

    result = {

        "target_column": str(target_name),

        "problem_type": problem_type,

        "suggested_problem_type": suggested_problem_type,

        "confidence": confidence,

        "requires_confirmation": bool(
            requires_confirmation
        ),

        "reason": reason,

        "target_statistics": {

            "total_records": int(len(series)),

            "valid_records": int(len(clean_series)),

            "missing_values": int(series.isna().sum()),

            "unique_values": int(
                clean_series.nunique()
            ),

            "data_type": str(series.dtype)

        },

        "class_analysis": None,

        "numeric_analysis": None,

        "warnings": warnings or []

    }

    if problem_type in (
        "binary_classification",
        "multiclass_classification"
    ):

        result["class_analysis"] = (
            get_target_distribution(clean_series)
        )

    if is_numeric_dtype(clean_series):

        result["numeric_analysis"] = (
            get_numeric_statistics(clean_series)
        )

    return result


# ==================================================
# MAIN FUNCTION: TARGET DETECTION
# ==================================================

def detect_target_type(
    series,
    target_name="target",
    user_choice=None
):
    """
    Automatically identify the ML problem type.

    Supported:
        - Binary classification
        - Multiclass classification
        - Regression
        - Ambiguous target

    user_choice:
        None
        "classification"
        "regression"
        "binary_classification"
        "multiclass_classification"

    The user's explicit choice overrides automatic
    inference, but invalid targets are still rejected.
    """

    # --------------------------------------------------
    # STEP 1: INPUT VALIDATION
    # --------------------------------------------------

    if not isinstance(series, pd.Series):

        raise TypeError(
            "Target must be a Pandas Series."
        )

    if series.empty:

        raise ValueError(
            "Target column is empty."
        )

    valid_choices = {
        None,
        "classification",
        "regression",
        "binary_classification",
        "multiclass_classification"
    }

    if user_choice not in valid_choices:

        raise ValueError(
            f"Invalid user_choice: {user_choice}"
        )

    # --------------------------------------------------
    # STEP 2: CHECK MISSING VALUES AND VALID DATA
    # --------------------------------------------------

    warnings = []

    if series.isna().any():

        warnings.append(
            "Target contains missing values. "
            "Rows with missing target values must be "
            "handled before model training."
        )

    clean_series = series.dropna()

    if clean_series.empty:

        return create_result(

            problem_type="invalid",

            suggested_problem_type=None,

            confidence="none",

            requires_confirmation=False,

            reason="Target contains no valid observations.",

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 3: CHECK INFINITE VALUES
    # --------------------------------------------------

    if is_numeric_dtype(clean_series):

        numeric_values = pd.to_numeric(
            clean_series,
            errors="coerce"
        )

        if not np.isfinite(
            numeric_values.to_numpy(dtype=float)
        ).all():

            warnings.append(
                "Target contains infinite or invalid "
                "numerical values."
            )

            return create_result(

                problem_type="invalid",

                suggested_problem_type=None,

                confidence="none",

                requires_confirmation=False,

                reason=(
                    "Target contains non-finite numerical values."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

    # --------------------------------------------------
    # STEP 4: CHECK NUMBER OF OBSERVATIONS
    # --------------------------------------------------

    if len(clean_series) < MIN_RECORDS:

        return create_result(

            problem_type="invalid",

            suggested_problem_type=None,

            confidence="none",

            requires_confirmation=False,

            reason=(
                "Insufficient valid observations "
                "for supervised learning."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 5: CHECK NUMBER OF UNIQUE VALUES
    # --------------------------------------------------

    unique_count = int(
        clean_series.nunique()
    )

    if unique_count < MIN_CLASSES:

        return create_result(

            problem_type="invalid",

            suggested_problem_type=None,

            confidence="none",

            requires_confirmation=False,

            reason=(
                "Target contains only one unique value. "
                "A predictive model cannot learn a "
                "meaningful distinction."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 6: HANDLE USER SELECTION
    # --------------------------------------------------

    if user_choice is not None:

        is_numeric = is_numeric_dtype(clean_series)

        if user_choice in (
            "regression",
        ) and not is_numeric:

            raise ValueError(
                "Regression requires a numerical target."
            )

        if user_choice in (
            "binary_classification",
        ) and unique_count != 2:

            raise ValueError(
                "Binary classification requires exactly "
                "two target classes."
            )

        if user_choice in (
            "multiclass_classification",
        ) and unique_count < 3:

            raise ValueError(
                "Multiclass classification requires "
                "at least three target classes."
            )

        if user_choice == "regression":

            return create_result(

                problem_type="regression",

                suggested_problem_type="regression",

                confidence="user_confirmed",

                requires_confirmation=False,

                reason=(
                    "User explicitly selected regression."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

        if user_choice == "binary_classification":

            return create_result(

                problem_type="binary_classification",

                suggested_problem_type="binary_classification",

                confidence="user_confirmed",

                requires_confirmation=False,

                reason=(
                    "User explicitly selected binary classification."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

        if user_choice == "multiclass_classification":

            return create_result(

                problem_type="multiclass_classification",

                suggested_problem_type="multiclass_classification",

                confidence="user_confirmed",

                requires_confirmation=False,

                reason=(
                    "User explicitly selected multiclass classification."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

        if user_choice == "classification":

            if unique_count == 2:

                selected_type = "binary_classification"

            else:

                selected_type = "multiclass_classification"

            return create_result(

                problem_type=selected_type,

                suggested_problem_type=selected_type,

                confidence="user_confirmed",

                requires_confirmation=False,

                reason=(
                    "User explicitly selected classification."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

    # --------------------------------------------------
    # STEP 7: DETECT BOOLEAN TARGET
    # --------------------------------------------------

    if is_bool_dtype(clean_series):

        return create_result(

            problem_type="binary_classification",

            suggested_problem_type="binary_classification",

            confidence="high",

            requires_confirmation=False,

            reason=(
                "Boolean target contains two possible "
                "outcomes and is suitable for binary classification."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 8: DETECT NON-NUMERICAL TARGET
    # --------------------------------------------------

    if not is_numeric_dtype(clean_series):

        if is_numeric_string(clean_series):

            warnings.append(
                "Target contains numerical-looking strings. "
                "Confirm whether these represent numeric "
                "measurements or category labels."
            )

            return create_result(

                problem_type="ambiguous",

                suggested_problem_type=None,

                confidence="low",

                requires_confirmation=True,

                reason=(
                    "Numeric-looking strings can represent "
                    "either numerical values or categorical labels."
                ),

                target_name=target_name,

                series=series,

                warnings=warnings

            )

        if unique_count == 2:

            selected_type = "binary_classification"

        else:

            selected_type = "multiclass_classification"

        return create_result(

            problem_type=selected_type,

            suggested_problem_type=selected_type,

            confidence="high",

            requires_confirmation=False,

            reason=(
                "Non-numerical target contains categorical "
                "labels. Classification is suggested."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 9: DETECT NUMERICAL TARGET
    # --------------------------------------------------

    if unique_count == 2:

        return create_result(

            problem_type="binary_classification",

            suggested_problem_type="binary_classification",

            confidence="medium",

            requires_confirmation=True,

            reason=(
                "Numerical target has exactly two unique values. "
                "It may represent binary classification or "
                "a numerical measurement."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 10: HANDLE LOW-CARDINALITY INTEGER TARGET
    # --------------------------------------------------

    if (
        pd.api.types.is_integer_dtype(clean_series)
        and unique_count <= MAX_CLASSES_FOR_AMBIGUOUS_INTEGER
    ):

        warnings.append(
            "Integer target has a small number of unique values. "
            "It could represent categories, ordinal ratings "
            "or numerical measurements."
        )

        return create_result(

            problem_type="ambiguous",

            suggested_problem_type=None,

            confidence="low",

            requires_confirmation=True,

            reason=(
                "Integer target has limited unique values. "
                "Its semantic meaning cannot be reliably "
                "determined from data type alone."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 11: HANDLE HIGH-CARDINALITY NUMERICAL TARGET
    # --------------------------------------------------

    if unique_count >= HIGH_CARDINALITY_THRESHOLD:

        return create_result(

            problem_type="regression",

            suggested_problem_type="regression",

            confidence="medium",

            requires_confirmation=False,

            reason=(
                "Numerical target contains many distinct "
                "values. Regression is suggested."
            ),

            target_name=target_name,

            series=series,

            warnings=warnings

        )

    # --------------------------------------------------
    # STEP 12: HANDLE OTHER NUMERICAL TARGETS
    # --------------------------------------------------

    return create_result(

        problem_type="ambiguous",

        suggested_problem_type="regression",

        confidence="low",

        requires_confirmation=True,

        reason=(
            "Numerical target has a limited number of distinct "
            "values. Confirm whether the prediction task "
            "is classification or regression."
        ),

        target_name=target_name,

        series=series,

        warnings=warnings

    )

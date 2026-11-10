
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.preprocessing import FunctionTransformer


# ==================================================
# CONFIGURATION
# ==================================================

TREE_BASED_ALGORITHMS = {
    "decision_tree",
    "decision_tree_classifier",
    "decision_tree_regressor",
    "random_forest",
    "random_forest_classifier",
    "random_forest_regressor",
    "extra_trees",
    "extra_trees_classifier",
    "extra_trees_regressor",
    "gradient_boosting",
    "gradient_boosting_classifier",
    "gradient_boosting_regressor",
    "hist_gradient_boosting",
    "hist_gradient_boosting_classifier",
    "hist_gradient_boosting_regressor"
}

SCALING_ALGORITHMS = {
    "logistic_regression",
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "svm",
    "svc",
    "svr",
    "knn",
    "kneighbors",
    "kneighbors_classifier",
    "kneighbors_regressor",
    "mlp",
    "mlp_classifier",
    "mlp_regressor"
}


# ==================================================
# HELPER 1: VALIDATE INPUT DATA
# ==================================================

def validate_features(X):
    """
    Validate feature data before preprocessing.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a Pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature column names detected."
        )

    if len(X.columns) == 0:
        raise ValueError(
            "No feature columns available."
        )

    return X


# ==================================================
# HELPER 2: DETECT FEATURE TYPES
# ==================================================

def detect_feature_columns(X):
    """
    Separate features into numerical,
    categorical and boolean columns.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    boolean_columns = X.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Boolean columns must not be treated
    # as continuous numerical features.
    numeric_columns = [
        col for col in numeric_columns
        if col not in boolean_columns
    ]

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
        and col not in boolean_columns
        and col not in datetime_columns
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "boolean": boolean_columns,
        "datetime": datetime_columns
    }


def replace_infinite_values(X):
    X = np.asarray(X, dtype=float)

    return np.where(
        np.isfinite(X),
        X,
        np.nan
    )


# ==================================================
# HELPER 3: CREATE NUMERICAL PIPELINE
# ==================================================

def build_numeric_pipeline(scale_numeric=True):
    """
    Build numerical preprocessing pipeline.

    Steps:
        1. Convert infinite values to missing.
        2. Fill missing values using median.
        3. Apply scaling when required.
    """

    steps = [

        (
            "replace_infinity",
            FunctionTransformer(
                replace_infinite_values,
                validate=False
            )
        ),

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]

    if scale_numeric:

        steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    return Pipeline(steps)


# ==================================================
# HELPER 4: CREATE CATEGORICAL PIPELINE
# ==================================================

def create_one_hot_encoder():
    """
    Support both newer and older sklearn versions.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )


def build_categorical_pipeline():
    """
    Handle missing categorical values
    and encode categorical features.
    """

    return Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__"
            )
        ),

        (
            "encoder",
            create_one_hot_encoder()
        )

    ])


def normalize_boolean_values(X):
    """
    Convert Boolean values into strings while
    preserving missing values and object dtype.
    """

    X = np.asarray(X, dtype=object)

    result = np.empty(X.shape, dtype=object)

    for index, value in np.ndenumerate(X):

        if pd.isna(value):
            result[index] = np.nan

        elif isinstance(value, (bool, np.bool_)):
            result[index] = str(value)

        else:
            result[index] = str(value)

    return result

# ==================================================
# HELPER 5: CREATE BOOLEAN PIPELINE
# ==================================================

def build_boolean_pipeline():
    """
    Convert boolean features into categorical strings,
    impute missing values and apply one-hot encoding.
    """

    return Pipeline([

        (
            "normalize",
            FunctionTransformer(
                normalize_boolean_values,
                validate=False
            )
        ),

        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__"
            )
        ),

        (
            "encoder",
            create_one_hot_encoder()
        )

    ])
# ==================================================
# HELPER 6: DETECT SCALING REQUIREMENT
# ==================================================

def should_scale(algorithm):
    """
    Determine whether numerical scaling
    is appropriate for an algorithm.
    """

    algorithm = algorithm.lower().strip()

    if algorithm in TREE_BASED_ALGORITHMS:
        return False

    if algorithm in SCALING_ALGORITHMS:
        return True

    # Conservative default:
    # Scale unknown algorithms.
    return True


# ==================================================
# MAIN FUNCTION: BUILD PREPROCESSOR
# ==================================================

def build_preprocessor(
    X,
    algorithm="random_forest",
    scale_numeric=None
):
    """
    Automatically construct a preprocessing pipeline.

    Parameters:
        X:
            Training feature DataFrame.

        algorithm:
            Name of the ML algorithm.

        scale_numeric:
            True  -> Apply StandardScaler.
            False -> Do not scale.
            None  -> Determine from algorithm.

    Returns:
        sklearn ColumnTransformer
    """

    # ----------------------------------------------
    # STEP 1: VALIDATE DATA
    # ----------------------------------------------

    X = validate_features(X)

    # ----------------------------------------------
    # STEP 2: DETECT FEATURE TYPES
    # ----------------------------------------------

    feature_columns = detect_feature_columns(X)

    numeric_columns = feature_columns["numeric"]

    categorical_columns = feature_columns["categorical"]

    boolean_columns = feature_columns["boolean"]

    datetime_columns = feature_columns["datetime"]

    # ----------------------------------------------
    # STEP 3: HANDLE DATETIME COLUMNS
    # ----------------------------------------------

    if datetime_columns:

        raise ValueError(
            "Datetime columns require date feature "
            "engineering before preprocessing. "
            f"Unsupported columns: {datetime_columns}"
        )

    # ----------------------------------------------
    # STEP 4: DETERMINE SCALING
    # ----------------------------------------------

    if scale_numeric is None:

        scale_numeric = should_scale(algorithm)

    # ----------------------------------------------
    # STEP 5: CREATE TRANSFORMERS
    # ----------------------------------------------

    transformers = []

    if numeric_columns:

        transformers.append(

            (
                "numeric",
                build_numeric_pipeline(
                    scale_numeric=scale_numeric
                ),
                numeric_columns
            )

        )

    if categorical_columns:

        transformers.append(

            (
                "categorical",
                build_categorical_pipeline(),
                categorical_columns
            )

        )

    if boolean_columns:

        transformers.append(

            (
                "boolean",
                build_boolean_pipeline(),
                boolean_columns
            )

        )

    # ----------------------------------------------
    # STEP 6: VALIDATE TRANSFORMERS
    # ----------------------------------------------

    if not transformers:

        raise ValueError(
            "No supported feature columns found."
        )

    # ----------------------------------------------
    # STEP 7: BUILD COLUMN TRANSFORMER
    # ----------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop",

        verbose_feature_names_out=False

    )

    return preprocessor


# ==================================================
# HELPER 7: GET FEATURE INFORMATION
# ==================================================

def get_preprocessing_summary(
    X,
    algorithm="random_forest"
):
    """
    Return preprocessing configuration
    for the diagnostic report.
    """

    X = validate_features(X)

    columns = detect_feature_columns(X)

    scale_numeric = should_scale(algorithm)

    return {

        "algorithm": algorithm,

        "numeric_columns": columns["numeric"],

        "categorical_columns": columns["categorical"],

        "boolean_columns": columns["boolean"],

        "datetime_columns": columns["datetime"],

        "scaling_enabled": scale_numeric,

        "numeric_imputation": "median",

        "categorical_imputation": "constant_missing",

        "categorical_encoding": "one_hot_encoding",

        "unknown_categories": "ignored"

    }

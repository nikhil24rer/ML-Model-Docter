
"""
Model Registry
--------------
Central registry of supported machine learning algorithms.

Responsibilities:
1. Register classification and regression models.
2. Store model metadata.
3. Store default hyperparameters.
4. Create fresh model instances.
5. Provide model information to other modules.
"""

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, SVR

import inspect


# ==================================================
# 1. CLASSIFICATION MODELS
# ==================================================

CLASSIFICATION_MODELS = {

    "logistic_regression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression,
        "description": "Linear model for binary and multiclass classification.",
        "category": "linear",
        "requires_scaling": True,
        "supports_probability": True,
        "default_params": {
            "max_iter": 1000,
            "random_state": 42,
        },
    },

    "decision_tree": {
        "name": "Decision Tree Classifier",
        "estimator": DecisionTreeClassifier,
        "description": "Tree-based classification algorithm.",
        "category": "tree",
        "requires_scaling": False,
        "supports_probability": True,
        "default_params": {
            "max_depth": 15,
            "random_state": 42,
        },
    },

    "random_forest": {
        "name": "Random Forest Classifier",
        "estimator": RandomForestClassifier,
        "description": "Ensemble of decision trees.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": True,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },

    "extra_trees": {
        "name": "Extra Trees Classifier",
        "estimator": ExtraTreesClassifier,
        "description": "Ensemble of highly randomized decision trees.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": True,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },

    "gradient_boosting": {
        "name": "Gradient Boosting Classifier",
        "estimator": GradientBoostingClassifier,
        "description": "Sequential boosting-based classification model.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": True,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
        },
    },

    "knn": {
        "name": "K-Nearest Neighbors",
        "estimator": KNeighborsClassifier,
        "description": "Distance-based classification algorithm.",
        "category": "distance",
        "requires_scaling": True,
        "supports_probability": True,
        "default_params": {
            "n_neighbors": 5,
        },
    },

    "naive_bayes": {
        "name": "Gaussian Naive Bayes",
        "estimator": GaussianNB,
        "description": "Probabilistic classification algorithm.",
        "category": "probabilistic",
        "requires_scaling": False,
        "supports_probability": True,
        "default_params": {},
    },

    "svm": {
        "name": "Support Vector Machine",
        "estimator": SVC,
        "description": "Kernel-based classification algorithm.",
        "category": "kernel",
        "requires_scaling": True,
        "supports_probability": True,
        "default_params": {
            "probability": True,
            "random_state": 42,
        },
    },
}


# ==================================================
# 2. REGRESSION MODELS
# ==================================================

REGRESSION_MODELS = {

    "linear_regression": {
        "name": "Linear Regression",
        "estimator": LinearRegression,
        "description": "Linear model for continuous value prediction.",
        "category": "linear",
        "requires_scaling": True,
        "supports_probability": False,
        "default_params": {},
    },

    "ridge": {
        "name": "Ridge Regression",
        "estimator": Ridge,
        "description": "Regularized linear regression algorithm.",
        "category": "linear",
        "requires_scaling": True,
        "supports_probability": False,
        "default_params": {
            "alpha": 1.0,
        },
    },

    "decision_tree_regressor": {
        "name": "Decision Tree Regressor",
        "estimator": DecisionTreeRegressor,
        "description": "Tree-based regression algorithm.",
        "category": "tree",
        "requires_scaling": False,
        "supports_probability": False,
        "default_params": {
            "max_depth": 15,
            "random_state": 42,
        },
    },

    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "estimator": RandomForestRegressor,
        "description": "Ensemble of regression trees.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": False,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },

    "extra_trees_regressor": {
        "name": "Extra Trees Regressor",
        "estimator": ExtraTreesRegressor,
        "description": "Ensemble of randomized regression trees.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": False,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },

    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "estimator": GradientBoostingRegressor,
        "description": "Sequential boosting-based regression model.",
        "category": "ensemble",
        "requires_scaling": False,
        "supports_probability": False,
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
        },
    },

    "knn_regressor": {
        "name": "K-Nearest Neighbors Regressor",
        "estimator": KNeighborsRegressor,
        "description": "Distance-based regression algorithm.",
        "category": "distance",
        "requires_scaling": True,
        "supports_probability": False,
        "default_params": {
            "n_neighbors": 5,
        },
    },

    "svr": {
        "name": "Support Vector Regressor",
        "estimator": SVR,
        "description": "Kernel-based regression algorithm.",
        "category": "kernel",
        "requires_scaling": True,
        "supports_probability": False,
        "default_params": {
            "kernel": "rbf",
        },
    },
}


# ==================================================
# 3. GET REGISTRY BY PROBLEM TYPE
# ==================================================

def get_model_registry(problem_type):
    """
    Return registered models for the given problem type.
    """

    if problem_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        return CLASSIFICATION_MODELS.copy()

    if problem_type == "regression":
        return REGRESSION_MODELS.copy()

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


# ==================================================
# 4. CREATE MODEL INSTANCE
# ==================================================

def create_model(model_name, problem_type, **model_params):
    """
    Create a fresh model instance.

    Registered defaults are used unless overridden.

    Unsupported random_state parameters are ignored
    for backward compatibility with the existing engine.
    """

    registry = get_model_registry(problem_type)

    if model_name not in registry:
        raise ValueError(
            f"Model '{model_name}' is not registered "
            f"for '{problem_type}'."
        )

    model_info = registry[model_name]

    estimator_class = model_info["estimator"]

    params = model_info["default_params"].copy()

    # User-supplied values override defaults.
    params.update(model_params)

    # Keep only the compatibility adjustment needed by
    # the current model_engine.py.
    supported_parameters = inspect.signature(
        estimator_class
    ).parameters

    if (
        "random_state" in params
        and "random_state" not in supported_parameters
    ):
        params.pop("random_state")

    return estimator_class(**params)


# ==================================================
# 5. GET MODEL INFORMATION
# ==================================================

def get_model_info(model_name, problem_type):
    """
    Return metadata for one model.
    """

    registry = get_model_registry(problem_type)

    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'."
        )

    details = registry[model_name]

    return {
        "id": model_name,
        "name": details["name"],
        "description": details["description"],
        "category": details["category"],
        "requires_scaling": details["requires_scaling"],
        "supports_probability": details["supports_probability"],
    }


# ==================================================
# 6. GET ALL AVAILABLE MODELS
# ==================================================

def get_available_models(problem_type):
    """
    Return serializable metadata for all registered models.
    """

    registry = get_model_registry(problem_type)

    return [
        get_model_info(model_name, problem_type)
        for model_name in registry
    ]

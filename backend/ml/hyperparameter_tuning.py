
from typing import Any, Dict, Optional

import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. VALIDATION HELPERS
# ============================================================

def validate_tuning_inputs(
    model,
    X,
    y,
    param_grid,
    problem_type,
    cv,
):
    """Validate model tuning inputs."""

    if model is None:
        raise ValueError("A model or pipeline is required.")

    if X is None or y is None:
        raise ValueError("X and y are required.")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    if param_grid is None or not isinstance(param_grid, dict):
        raise ValueError("param_grid must be a dictionary.")

    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    if problem_type not in ("classification", "regression"):
        raise ValueError(
            "problem_type must be classification or regression."
        )

    if not isinstance(cv, int) or cv < 2:
        raise ValueError("cv must be an integer greater than or equal to 2.")

    if len(X) < cv:
        raise ValueError("Number of samples must be at least equal to cv.")

    if problem_type == "classification":
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        if counts.min() < cv:
            raise ValueError(
                "Each class must have at least cv samples for stratified CV."
            )


# ============================================================
# 2. CROSS-VALIDATION STRATEGY
# ============================================================

def build_cv_strategy(problem_type, y, cv=5, random_state=42):
    """Build a suitable cross-validation strategy."""

    if problem_type == "classification":
        return StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

    return KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )


# ============================================================
# 3. DEFAULT SCORING
# ============================================================

def get_default_scoring(problem_type):
    """Return a default optimization metric."""

    if problem_type == "classification":
        return "f1_macro"

    if problem_type == "regression":
        return "r2"

    raise ValueError("Unsupported problem type.")


# ============================================================
# 4. NORMALIZE PARAMETER GRID
# ============================================================

def normalize_param_grid(model, param_grid):
    """
    Add pipeline step prefixes when needed.

    For a Pipeline whose final step is named 'model',
    a parameter such as C becomes model__C.
    """

    if not isinstance(model, Pipeline):
        return param_grid

    if not model.steps:
        raise ValueError("Pipeline must contain at least one step.")

    final_step_name = model.steps[-1][0]
    final_estimator = model.steps[-1][1]

    valid_params = final_estimator.get_params(deep=True)

    normalized = {}

    for key, values in param_grid.items():

        if "__" in key:
            normalized[key] = values

        elif key in valid_params:
            normalized[f"{final_step_name}__{key}"] = values

        else:
            raise ValueError(
                f"Unknown parameter '{key}' for the final pipeline estimator."
            )

    return normalized


# ============================================================
# 5. BUILD SEARCH REPORT
# ============================================================

def build_search_report(search, search_method, scoring):
    """Convert sklearn search results into a report."""

    best_index = search.best_index_
    cv_results = search.cv_results_

    best_std = float(
        cv_results["std_test_score"][best_index]
    )

    best_score = float(search.best_score_)

    results = []

    for index, params in enumerate(cv_results["params"]):

        results.append({
            "rank": int(cv_results["rank_test_score"][index]),
            "params": params,
            "mean_cv_score": float(
                cv_results["mean_test_score"][index]
            ),
            "std_cv_score": float(
                cv_results["std_test_score"][index]
            ),
        })

    results.sort(key=lambda item: item["rank"])

    return {
        "search_method": search_method,
        "scoring": scoring,
        "best_score": best_score,
        "best_score_std": best_std,
        "best_params": search.best_params_,
        "total_configurations_tested": len(results),
        "best_estimator": search.best_estimator_,
        "results": results,
        "interpretation": (
            "Best score is a cross-validation estimate. "
            "Use an untouched test set for final evaluation."
        ),
    }


# ============================================================
# 6. GRID SEARCH
# ============================================================

def tune_with_grid_search(
    model,
    X,
    y,
    param_grid,
    problem_type="classification",
    scoring=None,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    error_score="raise",
):
    """Tune model using exhaustive GridSearchCV."""

    validate_tuning_inputs(
        model,
        X,
        y,
        param_grid,
        problem_type,
        cv,
    )

    scoring = scoring or get_default_scoring(problem_type)

    cv_strategy = build_cv_strategy(
        problem_type,
        y,
        cv,
        random_state,
    )

    normalized_grid = normalize_param_grid(
        model,
        param_grid,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=normalized_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        error_score=error_score,
        return_train_score=True,
    )

    search.fit(X, y)

    return build_search_report(
        search,
        "grid_search",
        scoring,
    )


# ============================================================
# 7. RANDOMIZED SEARCH
# ============================================================

def tune_with_random_search(
    model,
    X,
    y,
    param_distributions,
    problem_type="classification",
    scoring=None,
    cv=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    error_score="raise",
):
    """Tune model using RandomizedSearchCV."""

    validate_tuning_inputs(
        model,
        X,
        y,
        param_distributions,
        problem_type,
        cv,
    )

    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer.")

    scoring = scoring or get_default_scoring(problem_type)

    cv_strategy = build_cv_strategy(
        problem_type,
        y,
        cv,
        random_state,
    )

    normalized_distributions = normalize_param_grid(
        model,
        param_distributions,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=normalized_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=verbose,
        error_score=error_score,
        return_train_score=True,
    )

    search.fit(X, y)

    return build_search_report(
        search,
        "randomized_search",
        scoring,
    )


# ============================================================
# 8. UNIFIED TUNING FUNCTION
# ============================================================

def tune_model(
    model,
    X,
    y,
    param_grid,
    problem_type="classification",
    method="grid",
    scoring=None,
    cv=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
):
    """
    Unified entry point for model hyperparameter tuning.

    method:
        grid
        random
    """

    if method == "grid":

        return tune_with_grid_search(
            model=model,
            X=X,
            y=y,
            param_grid=param_grid,
            problem_type=problem_type,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if method == "random":

        return tune_with_random_search(
            model=model,
            X=X,
            y=y,
            param_distributions=param_grid,
            problem_type=problem_type,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(
        "method must be either 'grid' or 'random'."
    )


# ============================================================
# 9. EXTRACT BEST MODEL
# ============================================================

def get_best_model(tuning_report):
    """Extract the fitted best estimator."""

    if not isinstance(tuning_report, dict):
        raise ValueError("Invalid tuning report.")

    best_model = tuning_report.get("best_estimator")

    if best_model is None:
        raise ValueError("No fitted best estimator found.")

    return best_model

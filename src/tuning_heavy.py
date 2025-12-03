import warnings
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Optional: silence the annoying label encoder warnings from xgboost
warnings.filterwarnings("ignore", category=UserWarning)


def tune_xgboost_grid_heavy(
    X, y
) -> Tuple[Dict[str, Any], float, GridSearchCV]:
    """
    Heavy XGBoost hyperparameter tuning using GridSearchCV.

    - Uses 5-fold StratifiedKFold CV
    - Large param grid (many combinations)
    - Optimizes ROC-AUC
    """

    print("=" * 80)
    print("XGBOOST GRID SEARCH – HEAVY MODE START")
    print("=" * 80)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # HEAVY grid – many combinations, this is intentionally large
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.003, 0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800, 1200],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # Base model with fixed things that we are NOT tuning
    base_model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",       # fast histogram-based
        eval_metric="auc",        # AUC matches our scoring
        random_state=42,
        n_jobs=-1,
        # DO NOT pass "use_label_encoder" any more
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,        # use all cores
        verbose=2,        # show progress
        refit=True,       # keep best model fitted on whole dataset
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print()
    print("Best XGBoost params (HEAVY GRID):")
    print(best_params)
    print(f"Best XGBoost ROC-AUC (CV): {best_score:.4f}")
    print("XGBOOST GRID SEARCH – HEAVY MODE DONE")
    print("=" * 80)
    print()

    return best_params, best_score, grid_search


def _catboost_objective(
    trial: optuna.Trial,
    X,
    y,
    cv: StratifiedKFold,
) -> float:
    """
    Optuna objective function for CatBoost – HEAVY tuning.
    """

    # Search space – intentionally wide / expensive
    iterations = trial.suggest_int("iterations", 300, 1200)
    depth = trial.suggest_int("depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
    border_count = trial.suggest_int("border_count", 32, 255)

    # CatBoostClassifier with Optuna-chosen hyperparams
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        random_state=42,
        verbose=False,  # keep logs clean; CV will run this many times
        thread_count=-1,
    )

    # 5-fold CV with ROC-AUC
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Optuna maximizes objective, so return mean AUC
    return float(np.mean(scores))


def tune_catboost_optuna_heavy(
    X,
    y,
    n_trials: int = 200,
) -> Tuple[Dict[str, Any], float]:
    """
    Heavy CatBoost hyperparameter tuning using Optuna.

    - 5-fold StratifiedKFold CV
    - Many trials (n_trials = 200 by default)
    - Optimizes ROC-AUC
    """

    print("=" * 80)
    print("CATBOOST OPTUNA TUNING – HEAVY MODE START")
    print("=" * 80)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        lambda trial: _catboost_objective(trial, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_score = float(study.best_value)

    print()
    print("Best CatBoost params (HEAVY OPTUNA):")
    print(best_params)
    print(f"Best CatBoost ROC-AUC (CV): {best_score:.4f}")
    print("CATBOOST OPTUNA TUNING – HEAVY MODE DONE")
    print("=" * 80)
    print()

    return best_params, best_score

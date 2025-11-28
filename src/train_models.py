import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna


def train_and_evaluate_models(X, y, feature_names):
    """
    Baseline training of XGBoost, LightGBM, CatBoost without tuning.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")

    models = {}

    # ---------------- XGBoost ----------------
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    # ---------------- LightGBM ----------------
    lgbm_model = LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
    lgbm_model.fit(X_train, y_train)
    models["LightGBM"] = lgbm_model

    # ---------------- CatBoost ----------------
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
    )
    cat_model.fit(X_train, y_train)
    models["CatBoost"] = cat_model

    # ---------------- Evaluation ----------------
    for name, model in models.items():
        print("\n" + "=" * 80)
        print(f"MODEL: {name}")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(
            model, "predict_proba"
        ) else None

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                print(f"ROC-AUC: {auc:.4f}")
            except Exception:
                print("ROC-AUC: could not be computed")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        if hasattr(model, "feature_importances_"):
            print("\nFeature Importances:")
            importances = model.feature_importances_
            for fn, imp in sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            ):
                print(f"{fn:40s} {imp:.4f}")

    return models


# =====================================================================
#  Hyperparameter tuning – XGBoost with GridSearchCV
# =====================================================================

def tune_xgboost_gridsearch(X, y):
    """
    Run a GridSearchCV on XGBoost to optimize ROC-AUC.

    This can take several minutes depending on your CPU.
    If you want to make it heavier, expand the param_grid.
    """

    print("\n" + "=" * 80)
    print("XGBOOST GRID SEARCH – START")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
    )

    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X, y)

    print("\nBest XGBoost params:")
    print(grid_search.best_params_)
    print(f"Best XGBoost ROC-AUC (CV): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    print("XGBOOST GRID SEARCH – DONE")
    print("=" * 80)

    return best_model, grid_search.best_params_, grid_search.best_score_


# =====================================================================
#  Hyperparameter tuning – CatBoost with Optuna
# =====================================================================

def _catboost_objective(trial, X, y):
    """
    Optuna objective for CatBoost. We do stratified 3-fold CV and maximize ROC-AUC.
    """

    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": 42,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
    }

    model = CatBoostClassifier(**params)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    return scores.mean()


def tune_catboost_optuna(X, y, n_trials: int = 30):
    """
    Run Optuna study to tune CatBoost hyperparameters.

    n_trials = how many combinations it will try.
    30 is okay. 100+ will start feeling heavy.
    """

    print("\n" + "=" * 80)
    print("CATBOOST OPTUNA TUNING – START")

    def objective(trial):
        return _catboost_objective(trial, X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest CatBoost params:")
    print(study.best_params)
    print(f"Best CatBoost ROC-AUC (CV): {study.best_value:.4f}")

    # Train final model on full data with best params
    best_params = study.best_params.copy()
    best_params.update(
        {
            "random_seed": 42,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
        }
    )
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X, y)

    print("CATBOOST OPTUNA TUNING – DONE")
    print("=" * 80)

    return best_model, study.best_params, study.best_value

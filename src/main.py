import os
import json

from data_loader import load_f1_data
from feature_engineering import build_feature_table, get_features_and_target

# Baseline fast models
from train_models import (
    train_and_evaluate_models,
)

# Heavy tuning imports
from tuning_heavy import (
    tune_xgboost_grid_heavy,
    tune_catboost_optuna_heavy,
)

import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def main():
    # =======================
    # 1. Load all F1 CSV files
    # =======================
    (
        races,
        results,
        drivers,
        constructors,
        qualifying,
        lap_times,
        status,
        circuits,
        seasons,
        driver_standings,
        constructor_standings,
        pit_stops,
        sprint_results,
        constructor_results,
    ) = load_f1_data()

    # =======================
    # 2. Build the full feature table
    # =======================
    df = build_feature_table(
        races,
        results,
        drivers,
        constructors,
        qualifying,
        lap_times,
        status,
        circuits,
        seasons,
        driver_standings,
        constructor_standings,
        pit_stops,
        sprint_results,
        constructor_results,
    )

    print(f"Total rows in final training table: {len(df)}")

    # =======================
    # 3. Extract features + target
    # =======================
    X, y, feature_names = get_features_and_target(df)

    # =======================
    # 4. Baseline models (fast)
    # =======================
    print("\n" + "=" * 80)
    print("BASELINE MODEL TRAINING (FAST)")
    print("=" * 80)

    # This should return a dict:
    # {
    #   "xgboost": xgb_model,
    #   "lightgbm": lgb_model,
    #   "catboost": cat_model
    # }
    baseline_models = train_and_evaluate_models(X, y, feature_names)

    # =======================
    # 5. HEAVY XGBoost tuning
    # =======================
    print("\n" + "=" * 80)
    print("HEAVY TUNING – XGBoost (GridSearchCV)")
    print("=" * 80)

    # tune_xgboost_grid_heavy must return:
    #   best_params: dict
    #   best_auc: float
    #   grid_obj: fitted GridSearchCV
    xgb_best_params, xgb_best_auc, xgb_grid = tune_xgboost_grid_heavy(X, y)

    # =======================
    # 6. HEAVY CatBoost tuning
    # =======================
    print("\n" + "=" * 80)
    print("HEAVY TUNING – CatBoost (Optuna)")
    print("=" * 80)

    # tune_catboost_optuna_heavy must return:
    #   best_params: dict
    #   best_auc: float
    cat_best_params, cat_best_auc = tune_catboost_optuna_heavy(
        X,
        y,
        n_trials=200,   # heavy search
    )

    # =======================
    # 7. Final Summary (tuning)
    # =======================
    print("\n" + "=" * 80)
    print("SUMMARY OF HEAVY TUNING RESULTS")
    print("=" * 80)
    print(f"Best XGBoost ROC-AUC (5-fold CV): {xgb_best_auc:.4f}")
    print(f"Best XGBoost Params: {xgb_best_params}")
    print("-" * 80)
    print(f"Best CatBoost ROC-AUC (5-fold CV, 200 trials): {cat_best_auc:.4f}")
    print(f"Best CatBoost Params: {cat_best_params}")
    print("=" * 80)

    # =======================
    # 8. SAVE MODELS + FEATURE NAMES
    # =======================
    os.makedirs("models", exist_ok=True)

    # 8.1 Save baseline models (if returned)
    if isinstance(baseline_models, dict):
        baseline_xgb = baseline_models.get("xgboost")
        baseline_lgbm = baseline_models.get("lightgbm")
        baseline_cat = baseline_models.get("catboost")
    else:
        baseline_xgb = baseline_lgbm = baseline_cat = None

    if baseline_xgb is not None:
        joblib.dump(baseline_xgb, os.path.join("models", "xgboost_baseline.pkl"))
        print("Saved baseline XGBoost model to models/xgboost_baseline.pkl")

    if baseline_lgbm is not None:
        joblib.dump(baseline_lgbm, os.path.join("models", "lightgbm_baseline.pkl"))
        print("Saved baseline LightGBM model to models/lightgbm_baseline.pkl")

    if baseline_cat is not None:
        joblib.dump(baseline_cat, os.path.join("models", "catboost_baseline.pkl"))
        print("Saved baseline CatBoost model to models/catboost_baseline.pkl")

    # 8.2 Save HEAVY-TUNED XGBoost model
    # Option 1: directly from GridSearchCV best_estimator_
    xgb_best_model = xgb_grid.best_estimator_

    # Option 2 (alternative) would be re-fitting a fresh XGBClassifier with best params,
    # but best_estimator_ is already fitted on the full data in GridSearchCV with refit=True.

    joblib.dump(xgb_best_model, os.path.join("models", "xgboost_heavy_best.pkl"))
    print("Saved heavy-tuned XGBoost model to models/xgboost_heavy_best.pkl")

    # 8.3 Train + save HEAVY-TUNED CatBoost model on full data
    # We re-fit CatBoost on the whole dataset using the best params from Optuna.
    cat_best_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
        **cat_best_params,
    )
    cat_best_model.fit(X, y)

    # You can either use CatBoost's own save_model or joblib. Using joblib for consistency.
    joblib.dump(cat_best_model, os.path.join("models", "catboost_heavy_best.pkl"))
    print("Saved heavy-tuned CatBoost model to models/catboost_heavy_best.pkl")

    # 8.4 Save feature names (for Flask / inference)
    feature_names_path = os.path.join("models", "feature_names.json")
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    print(f"Saved feature names to {feature_names_path}")

    print("\nAll models and feature names saved under the 'models/' directory.")
    print("Training complete.")


if __name__ == "__main__":
    main()

from data_loader import load_f1_data
from feature_engineering import build_feature_table, get_features_and_target
from train_models import (
    train_and_evaluate_models,
    tune_xgboost_gridsearch,
    tune_catboost_optuna,
)


def main():
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

    X, y, feature_names = get_features_and_target(df)

    # 1) Baseline models (fast)
    models = train_and_evaluate_models(X, y, feature_names)

    # 2) Hyperparameter tuning – XGBoost with GridSearchCV
    #    This will take longer than baseline training.
    best_xgb_model, best_xgb_params, best_xgb_auc = tune_xgboost_gridsearch(X, y)

    # 3) Hyperparameter tuning – CatBoost with Optuna
    #    Increase n_trials if you want to make it heavier.
    best_cat_model, best_cat_params, best_cat_auc = tune_catboost_optuna(
        X, y, n_trials=30
    )

    print("\n" + "=" * 80)
    print("SUMMARY OF TUNING RESULTS")
    print(f"Best XGBoost ROC-AUC (CV): {best_xgb_auc:.4f}")
    print(f"Best XGBoost Params: {best_xgb_params}")
    print(f"Best CatBoost ROC-AUC (CV): {best_cat_auc:.4f}")
    print(f"Best CatBoost Params: {best_cat_params}")
    print("=" * 80)


if __name__ == "__main__":
    main()

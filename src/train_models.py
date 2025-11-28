import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def train_and_evaluate_models(X, y, feature_names):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")

    models = {}

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

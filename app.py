import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------------
# 0. Import project modules from src/
# -----------------------------------------------------------------------------------
from src.data_loader import load_f1_data
from src.feature_engineering import build_feature_table

# -----------------------------------------------------------------------------------
# 1. Paths & model loading
# -----------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "f1")  # used by data_loader internally if needed

XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_heavy_best.pkl")
CAT_MODEL_PATH = os.path.join(MODELS_DIR, "catboost_heavy_best.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")


def load_feature_names(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_names.json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list):
        raise ValueError("feature_names.json must contain a JSON list of strings.")
    return feature_names


def load_xgb_model(path: str) -> XGBClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(f"XGBoost model file not found at: {path}")
    model = joblib.load(path)
    if not isinstance(model, XGBClassifier):
        raise TypeError("Loaded XGBoost model is not an instance of XGBClassifier.")
    return model


def load_catboost_model(path: str) -> CatBoostClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CatBoost model file not found at: {path}")
    model = joblib.load(path)
    if not isinstance(model, CatBoostClassifier):
        raise TypeError("Loaded CatBoost model is not an instance of CatBoostClassifier.")
    return model


# -----------------------------------------------------------------------------------
# 1.1 Load models + feature names at startup (fail fast if anything is broken)
# -----------------------------------------------------------------------------------

FEATURE_NAMES: List[str] = load_feature_names(FEATURE_NAMES_PATH)
XGB_MODEL: XGBClassifier = load_xgb_model(XGB_MODEL_PATH)
CAT_MODEL: CatBoostClassifier = load_catboost_model(CAT_MODEL_PATH)

# -----------------------------------------------------------------------------------
# 1.2 Load raw F1 data + build feature table once (for metadata / later extensions)
# -----------------------------------------------------------------------------------

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

FULL_FEATURE_DF: pd.DataFrame = build_feature_table(
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

# a small metadata table for the UI (drivers, races, seasons, circuits)
# you can tweak this as you like in the frontend
RACES_META = (
    races.merge(
        circuits[["circuitId", "name"]].rename(columns={"name": "circuit_name"}),
        on="circuitId",
        how="left",
    )
    .merge(
        seasons[["year"]],
        on="year",
        how="left",
    )
)

DRIVER_NAMES = (
    drivers["driverRef"].tolist()
    if "driverRef" in drivers.columns
    else drivers["surname"].tolist()
)

SEASONS_LIST = sorted(seasons["year"].unique().tolist())
CIRCUIT_NAMES = sorted(RACES_META["circuit_name"].dropna().unique().tolist())

# -----------------------------------------------------------------------------------
# 2. Helper functions to validate & convert incoming JSON to numpy arrays
# -----------------------------------------------------------------------------------

def validate_and_build_vector(payload: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
    """
    Convert a single JSON object into a 1D numpy array of shape (n_features,)
    following the exact order in feature_names.
    Raises ValueError if any feature is missing or not numeric.
    """
    values: List[float] = []

    for name in feature_names:
        if name not in payload:
            raise ValueError(f"Missing feature in request body: '{name}'")

        value = payload[name]

        # Allow int/float or numeric strings, but not nested structures
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, str):
            try:
                values.append(float(value))
            except ValueError:
                raise ValueError(f"Feature '{name}' must be numeric, got string '{value}'")
        else:
            raise ValueError(
                f"Feature '{name}' must be numeric (int/float/str), got type {type(value).__name__}"
            )

    return np.array(values, dtype=float)


def validate_and_build_matrix(
    payload_list: List[Dict[str, Any]],
    feature_names: List[str],
) -> np.ndarray:
    """
    Convert a list of JSON objects into a 2D numpy array of shape (n_samples, n_features).
    """
    if not isinstance(payload_list, list) or len(payload_list) == 0:
        raise ValueError("Request body must be a non-empty JSON array for batch prediction.")

    rows: List[np.ndarray] = []
    for idx, row in enumerate(payload_list):
        if not isinstance(row, dict):
            raise ValueError(f"Element at index {idx} must be a JSON object.")
        vec = validate_and_build_vector(row, feature_names)
        rows.append(vec)

    return np.vstack(rows)


# -----------------------------------------------------------------------------------
# 3. Flask app & endpoints
# -----------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow your Vite frontend (localhost) to call this API


@app.route("/health", methods=["GET"])
def health() -> Any:
    """
    Simple health check endpoint.
    """
    return jsonify(
        {
            "status": "ok",
            "models": {
                "xgboost_loaded": True,
                "catboost_loaded": True,
            },
            "n_features": len(FEATURE_NAMES),
            "feature_names_example": FEATURE_NAMES[:5],
            "rows_in_feature_table": int(len(FULL_FEATURE_DF)),
        }
    )


@app.route("/metadata", methods=["GET"])
def metadata() -> Any:
    """
    Return information that the frontend can use to build dropdowns:
    - list of driver names
    - list of seasons
    - list of circuits
    - basic race list (id, year, round, name, circuit)
    - full feature name list (for advanced users)
    """
    races_list = (
        RACES_META[
            ["raceId", "year", "round", "name", "circuit_name"]
        ]
        .sort_values(["year", "round"])
        .to_dict(orient="records")
    )

    return jsonify(
        {
            "drivers": DRIVER_NAMES,
            "seasons": SEASONS_LIST,
            "circuits": CIRCUIT_NAMES,
            "races": races_list,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        }
    )


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """
    Single prediction endpoint.

    Expect body:

    {
      "model": "catboost",   // optional: "catboost" (default) or "xgboost"
      "features": {
        "<feature_1>": value,
        "<feature_2>": value,
        ...
      }
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    model_name = data.get("model", "catboost")
    features_obj = data.get("features")

    if features_obj is None or not isinstance(features_obj, dict):
        return jsonify({"error": "Missing 'features' object in request body."}), 400

    try:
        x_vec = validate_and_build_vector(features_obj, FEATURE_NAMES)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    x_vec = x_vec.reshape(1, -1)

    if model_name.lower() == "xgboost":
        model = XGB_MODEL
        model_used = "xgboost_heavy_best"
    else:
        model = CAT_MODEL
        model_used = "catboost_heavy_best"

    try:
        proba = model.predict_proba(x_vec)[0, 1]
        pred_label = int(proba >= 0.5)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    return jsonify(
        {
            "model_used": model_used,
            "probability_win": float(proba),
            "predicted_label": pred_label,
            "threshold": 0.5,
        }
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch() -> Any:
    """
    Batch prediction endpoint.

    Request JSON:

    {
      "model": "catboost",  // optional
      "rows": [
        { "<feature_1>": v1, "<feature_2>": v2, ... },
        { "<feature_1>": v1, "<feature_2>": v2, ... }
      ]
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    model_name = data.get("model", "catboost")
    rows = data.get("rows")

    if rows is None:
        return jsonify({"error": "Missing 'rows' array in request body."}), 400

    try:
        x_mat = validate_and_build_matrix(rows, FEATURE_NAMES)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if model_name.lower() == "xgboost":
        model = XGB_MODEL
        model_used = "xgboost_heavy_best"
    else:
        model = CAT_MODEL
        model_used = "catboost_heavy_best"

    try:
        proba = model.predict_proba(x_mat)[:, 1]
        preds = (proba >= 0.5).astype(int)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    results = []
    for i in range(len(rows)):
        results.append(
            {
                "index": i,
                "probability_win": float(proba[i]),
                "predicted_label": int(preds[i]),
            }
        )

    return jsonify(
        {
            "model_used": model_used,
            "threshold": 0.5,
            "results": results,
        }
    )


# -----------------------------------------------------------------------------------
# 4. Main entry point
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    # For local dev only. Put behind gunicorn/nginx in real deployment.
    app.run(host="0.0.0.0", port=5000, debug=True)

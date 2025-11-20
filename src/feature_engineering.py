import pandas as pd
import numpy as np

def build_feature_table(races, results, drivers, constructors):
    races_small = races[["raceId", "year", "round", "circuitId"]].copy()
    results_small = results[
        ["resultId", "raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]
    ].copy()

    df = results_small.merge(races_small, on="raceId", how="left")

    df = df.dropna(subset=["positionOrder"]).copy()
    df["is_winner"] = (df["positionOrder"] == 1).astype(int)

    df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)
    df["driver_races_before"] = df.groupby(["driverId", "year"]).cumcount()
    df["driver_points_before"] = df.groupby(["driverId", "year"])["points"].cumsum().shift(fill_value=0)
    df["driver_wins_before"] = df.groupby(["driverId", "year"])["is_winner"].cumsum().shift(fill_value=0)

    df = df.sort_values(["constructorId", "year", "round"]).reset_index(drop=True)
    df["constructor_races_before"] = df.groupby(["constructorId", "year"]).cumcount()
    df["constructor_points_before"] = df.groupby(["constructorId", "year"])["points"].cumsum().shift(fill_value=0)
    df["constructor_wins_before"] = df.groupby(["constructorId", "year"])["is_winner"].cumsum().shift(fill_value=0)

    df = df.sort_values(["year", "round", "raceId", "driverId"]).reset_index(drop=True)

    df["driver_points_before"] = df["driver_points_before"].fillna(0)
    df["driver_wins_before"] = df["driver_wins_before"].fillna(0)
    df["constructor_points_before"] = df["constructor_points_before"].fillna(0)
    df["constructor_wins_before"] = df["constructor_wins_before"].fillna(0)

    return df


def get_features_and_target(df):
    feature_cols = [
        "year",
        "round",
        "circuitId",
        "grid",
        "driver_races_before",
        "driver_points_before",
        "driver_wins_before",
        "constructor_races_before",
        "constructor_points_before",
        "constructor_wins_before",
    ]

    X = df[feature_cols].astype(np.float32).values
    y = df["is_winner"].values
    return X, y, feature_cols

import pandas as pd
import numpy as np


def _time_str_to_ms(t):
    """Convert a qualifying time string like '1:23.456' to milliseconds."""
    if pd.isna(t):
        return np.nan
    if not isinstance(t, str):
        return np.nan
    t = t.strip()
    if not t:
        return np.nan
    parts = t.split(":")
    if len(parts) != 2:
        return np.nan
    try:
        minutes = int(parts[0])
        seconds = float(parts[1])
        total_seconds = minutes * 60.0 + seconds
        return int(total_seconds * 1000)
    except ValueError:
        return np.nan


def _is_dnf(status_str: str) -> int:
    """Very simple DNF classifier based on the 'status' text."""
    if pd.isna(status_str):
        return 0
    s = str(status_str).lower()
    if "finished" in s:
        return 0
    if "+" in s:
        return 0
    if "lap" in s:
        return 0
    return 1


def build_feature_table(
    races: pd.DataFrame,
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    qualifying: pd.DataFrame,
    lap_times: pd.DataFrame,
    status: pd.DataFrame,
    circuits: pd.DataFrame,
    seasons: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    pit_stops: pd.DataFrame,
    sprint_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-driver-per-race feature table using all Kaggle F1 CSVs.
    """

    # ---------- Circuits + race metadata ----------
    circuits_small = circuits[["circuitId", "lat", "lng", "alt", "country"]].copy()
    circuits_small["circuit_country_id"], _ = pd.factorize(circuits_small["country"])
    circuits_small = circuits_small.drop(columns=["country"])

    races_small = races[["raceId", "year", "round", "circuitId", "time"]].copy()
    races_small = races_small.merge(circuits_small, on="circuitId", how="left")

    def _is_night(time_val):
        if pd.isna(time_val):
            return 0
        s = str(time_val)
        try:
            hour = int(s.split(":")[0])
            return 1 if hour >= 18 else 0
        except Exception:
            return 0

    races_small["is_night_race"] = races_small["time"].apply(_is_night)
    races_small = races_small.drop(columns=["time"])

    seasons_sorted = seasons.sort_values("year").reset_index(drop=True)
    seasons_sorted["season_index"] = np.arange(1, len(seasons_sorted) + 1)
    races_small = races_small.merge(
        seasons_sorted[["year", "season_index"]], on="year", how="left"
    )

    # ---------- Base results + status ----------
    results_small = results[
        [
            "resultId",
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "positionOrder",
            "points",
            "statusId",
        ]
    ].copy()
    status_small = status[["statusId", "status"]].copy()

    df = results_small.merge(races_small, on="raceId", how="left")
    df = df.merge(status_small, on="statusId", how="left")

    # ---------- Qualifying ----------
    qual_small = qualifying[
        ["raceId", "driverId", "position", "q1", "q2", "q3"]
    ].copy()
    qual_small = qual_small.rename(columns={"position": "qual_position"})
    df = df.merge(qual_small, on=["raceId", "driverId"], how="left")

    # Drop invalid results
    df = df.dropna(subset=["positionOrder"]).copy()

    # Target + DNF flag
    df["is_winner"] = (df["positionOrder"] == 1).astype(int)
    df["is_dnf"] = df["status"].apply(_is_dnf)

    # ---------- Qualifying features ----------
    df["qual_position"] = df["qual_position"].fillna(df["grid"])
    global_qual_mean = df["qual_position"].mean()
    df["qual_position"] = df["qual_position"].fillna(global_qual_mean)

    for src, dst in [("q1", "q1_ms"), ("q2", "q2_ms"), ("q3", "q3_ms")]:
        df[dst] = df[src].apply(_time_str_to_ms)

    df["best_qual_ms"] = df[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)
    race_best = df.groupby("raceId")["best_qual_ms"].transform("min")
    df["gap_to_pole_ms"] = df["best_qual_ms"] - race_best
    df["gap_to_pole_ms"] = df["gap_to_pole_ms"].fillna(0)

    for col in ["q1_ms", "q2_ms", "q3_ms", "best_qual_ms"]:
        df[col] = df[col].fillna(df[col].mean())

    # ---------- Race pace from lap_times ----------
    lap = lap_times[["raceId", "driverId", "milliseconds"]].dropna().copy()
    lap = lap.merge(races_small[["raceId", "year", "round"]], on="raceId", how="left")

    lap_avg = (
        lap.groupby(["driverId", "year", "round"], as_index=False)["milliseconds"]
        .mean()
        .rename(columns={"milliseconds": "avg_lap_ms_this_race"})
    )
    global_pace_mean = lap_avg["avg_lap_ms_this_race"].mean()

    lap_avg = lap_avg.sort_values(["driverId", "year", "round"])
    lap_avg["driver_race_count"] = lap_avg.groupby(["driverId", "year"]).cumcount()
    lap_avg["pace_cumsum_before"] = (
        lap_avg.groupby(["driverId", "year"])["avg_lap_ms_this_race"]
        .cumsum()
        .shift(fill_value=np.nan)
    )
    lap_avg["driver_avg_pace_ms_before"] = np.where(
        lap_avg["driver_race_count"] > 0,
        lap_avg["pace_cumsum_before"] / lap_avg["driver_race_count"],
        global_pace_mean,
    )

    lap_feat = lap_avg[
        ["driverId", "year", "round", "driver_avg_pace_ms_before"]
    ].copy()
    df = df.merge(lap_feat, on=["driverId", "year", "round"], how="left")
    df["driver_avg_pace_ms_before"] = df["driver_avg_pace_ms_before"].fillna(
        global_pace_mean
    )

    # ---------- Pit stop strategy (pit_stops) ----------
    pit = pit_stops[["raceId", "driverId", "milliseconds"]].dropna().copy()
    pit = pit.merge(races_small[["raceId", "year", "round"]], on="raceId", how="left")

    pit_agg = (
        pit.groupby(["driverId", "year", "round"], as_index=False)["milliseconds"]
        .mean()
        .rename(columns={"milliseconds": "avg_pit_ms_this_race"})
    )
    pit_agg = pit_agg.sort_values(["driverId", "year", "round"])
    pit_agg["race_count"] = pit_agg.groupby(["driverId", "year"]).cumcount()
    pit_agg["pit_cumsum_before"] = (
        pit_agg.groupby(["driverId", "year"])["avg_pit_ms_this_race"]
        .cumsum()
        .shift(fill_value=np.nan)
    )
    global_pit_mean = pit_agg["avg_pit_ms_this_race"].mean()

    pit_agg["driver_avg_pit_ms_before"] = np.where(
        pit_agg["race_count"] > 0,
        pit_agg["pit_cumsum_before"] / pit_agg["race_count"],
        global_pit_mean,
    )

    pit_feat = pit_agg[
        ["driverId", "year", "round", "driver_avg_pit_ms_before"]
    ].copy()
    df = df.merge(pit_feat, on=["driverId", "year", "round"], how="left")
    df["driver_avg_pit_ms_before"] = df["driver_avg_pit_ms_before"].fillna(
        global_pit_mean
    )

    # ---------- Sprint results (driver sprint form) ----------
    if not sprint_results.empty:
        sprint = sprint_results[
            ["raceId", "driverId", "position", "points"]
        ].copy()
        sprint["position"] = pd.to_numeric(
            sprint["position"], errors="coerce"
        )
        sprint["points"] = pd.to_numeric(
            sprint["points"], errors="coerce"
        ).fillna(0)

        sprint = sprint.rename(
            columns={"position": "sprint_position", "points": "sprint_points"}
        )
        sprint = sprint.merge(
            races_small[["raceId", "year", "round"]], on="raceId", how="left"
        )

        sprint = sprint.sort_values(["driverId", "year", "round"])
        sprint["sprint_race_count"] = sprint.groupby(["driverId", "year"]).cumcount()
        sprint["sprint_points_cumsum_before"] = (
            sprint.groupby(["driverId", "year"])["sprint_points"]
            .cumsum()
            .shift(fill_value=0)
        )
        sprint["sprint_pos_cumsum_before"] = (
            sprint.groupby(["driverId", "year"])["sprint_position"]
            .cumsum()
            .shift(fill_value=0)
        )

        sprint["driver_sprint_points_before"] = sprint[
            "sprint_points_cumsum_before"
        ]
        sprint["driver_avg_sprint_pos_before"] = np.where(
            sprint["sprint_race_count"] > 0,
            sprint["sprint_pos_cumsum_before"] / sprint["sprint_race_count"],
            np.nan,
        )

        sprint_feat = sprint[
            [
                "driverId",
                "year",
                "round",
                "driver_sprint_points_before",
                "driver_avg_sprint_pos_before",
            ]
        ].copy()

        df = df.merge(
            sprint_feat,
            on=["driverId", "year", "round"],
            how="left",
        )
    else:
        df["driver_sprint_points_before"] = 0.0
        df["driver_avg_sprint_pos_before"] = np.nan

    sprint_race_ids = (
        sprint_results["raceId"].unique() if not sprint_results.empty else []
    )
    df["is_sprint_weekend"] = df["raceId"].isin(sprint_race_ids).astype(int)

    # ---------- Driver championship standings ----------
    ds = driver_standings.merge(
        races_small[["raceId", "year", "round"]], on="raceId", how="left"
    ).copy()
    ds["position"] = pd.to_numeric(ds["position"], errors="coerce")
    ds["points"] = pd.to_numeric(ds["points"], errors="coerce").fillna(0)

    ds = ds.sort_values(["driverId", "year", "round"])
    ds["champ_pos_before"] = (
        ds.groupby(["driverId", "year"])["position"].shift(1)
    )
    ds["champ_points_before"] = (
        ds.groupby(["driverId", "year"])["points"].cumsum().shift(fill_value=0)
    )

    ds_feat = ds[
        ["raceId", "driverId", "champ_pos_before", "champ_points_before"]
    ].copy()
    df = df.merge(ds_feat, on=["raceId", "driverId"], how="left")

    # ---------- Constructor championship standings ----------
    cs = constructor_standings.merge(
    races_small[["raceId", "year", "round"]], on="raceId", how="left"
    )

    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")
    cs["points"] = pd.to_numeric(cs["points"], errors="coerce").fillna(0)

    cs = cs.sort_values(["constructorId", "year", "round"])
    cs["const_champ_pos_before"] = (
        cs.groupby(["constructorId", "year"])["position"]
        .shift(1)
    )
    cs["const_champ_points_before"] = (
        cs.groupby(["constructorId", "year"])["points"].cumsum().shift(fill_value=0)
    )

    cs_feat = cs[
        ["raceId", "constructorId", "const_champ_pos_before", "const_champ_points_before"]
    ].copy()
    df = df.merge(cs_feat, on=["raceId", "constructorId"], how="left")

    # ---------- Constructor race points ----------
    cr = constructor_results[
        ["raceId", "constructorId", "points"]
    ].copy()
    cr["points"] = pd.to_numeric(cr["points"], errors="coerce").fillna(0)
    cr = cr.rename(columns={"points": "constructor_race_points"})
    df = df.merge(cr, on=["raceId", "constructorId"], how="left")

    # ---------- Driver-level rolling stats ----------
    df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)

    df["driver_races_before"] = df.groupby(["driverId", "year"]).cumcount()
    df["driver_points_before"] = (
        df.groupby(["driverId", "year"])["points"].cumsum().shift(fill_value=0)
    )
    df["driver_wins_before"] = (
        df.groupby(["driverId", "year"])["is_winner"].cumsum().shift(fill_value=0)
    )
    df["driver_dnf_before"] = (
        df.groupby(["driverId", "year"])["is_dnf"].cumsum().shift(fill_value=0)
    )
    df["driver_qual_pos_cumsum_before"] = (
        df.groupby(["driverId", "year"])["qual_position"].cumsum().shift(fill_value=0)
    )

    df["driver_avg_qual_pos_before"] = np.where(
        df["driver_races_before"] > 0,
        df["driver_qual_pos_cumsum_before"] / df["driver_races_before"],
        global_qual_mean,
    )

    global_dnf_rate = df["is_dnf"].mean()
    df["driver_dnf_rate_before"] = np.where(
        df["driver_races_before"] > 0,
        df["driver_dnf_before"] / df["driver_races_before"],
        global_dnf_rate,
    )

    # ---------- Constructor-level rolling stats ----------
    df = df.sort_values(["constructorId", "year", "round"]).reset_index(drop=True)

    df["constructor_races_before"] = (
        df.groupby(["constructorId", "year"]).cumcount()
    )
    df["constructor_points_before"] = (
        df.groupby(["constructorId", "year"])["points"].cumsum().shift(fill_value=0)
    )
    df["constructor_wins_before"] = (
        df.groupby(["constructorId", "year"])["is_winner"].cumsum().shift(fill_value=0)
    )
    df["constructor_dnf_before"] = (
        df.groupby(["constructorId", "year"])["is_dnf"].cumsum().shift(fill_value=0)
    )
    df["constructor_qual_pos_cumsum_before"] = (
        df.groupby(["constructorId", "year"])["qual_position"]
        .cumsum()
        .shift(fill_value=0)
    )

    df["constructor_avg_qual_pos_before"] = np.where(
        df["constructor_races_before"] > 0,
        df["constructor_qual_pos_cumsum_before"] / df["constructor_races_before"],
        global_qual_mean,
    )
    df["constructor_dnf_rate_before"] = np.where(
        df["constructor_races_before"] > 0,
        df["constructor_dnf_before"] / df["constructor_races_before"],
        global_dnf_rate,
    )

    # ---------- Final cleanup ----------
    df = df.sort_values(["year", "round", "raceId", "driverId"]).reset_index(drop=True)

    df["driver_sprint_points_before"] = df["driver_sprint_points_before"].fillna(0.0)
    df["driver_avg_sprint_pos_before"] = df["driver_avg_sprint_pos_before"].fillna(
        df["driver_avg_sprint_pos_before"].mean()
    )

    df["champ_pos_before"] = df["champ_pos_before"].fillna(
        df["champ_pos_before"].mean()
    )
    df["champ_points_before"] = df["champ_points_before"].fillna(0)

    df["const_champ_pos_before"] = df["const_champ_pos_before"].fillna(
        df["const_champ_pos_before"].mean()
    )
    df["const_champ_points_before"] = df["const_champ_points_before"].fillna(0)

    df["constructor_race_points"] = df["constructor_race_points"].fillna(0)

    return df


def get_features_and_target(df: pd.DataFrame):
    """Select final model features + target."""
    feature_cols = [
        "year",
        "season_index",
        "round",
        "circuitId",
        "lat",
        "lng",
        "alt",
        "circuit_country_id",
        "is_night_race",
        "is_sprint_weekend",
        "grid",
        "qual_position",
        "q1_ms",
        "q2_ms",
        "q3_ms",
        "best_qual_ms",
        "gap_to_pole_ms",
        "driver_races_before",
        "driver_points_before",
        "driver_wins_before",
        "driver_dnf_rate_before",
        "driver_avg_qual_pos_before",
        "driver_avg_pace_ms_before",
        "driver_avg_pit_ms_before",
        "driver_sprint_points_before",
        "driver_avg_sprint_pos_before",
        "champ_pos_before",
        "champ_points_before",
        "constructor_races_before",
        "constructor_points_before",
        "constructor_wins_before",
        "constructor_dnf_rate_before",
        "constructor_avg_qual_pos_before",
        "const_champ_pos_before",
        "const_champ_points_before",
        "constructor_race_points",
    ]

    X = df[feature_cols].astype(np.float32).values
    y = df["is_winner"].values
    return X, y, feature_cols

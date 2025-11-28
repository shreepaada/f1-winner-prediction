import pandas as pd


def load_f1_data(base_path: str = "data/f1/"):
    """
    Load ALL Formula 1 CSVs used by the project from the Kaggle dataset.
    """

    races = pd.read_csv(base_path + "races.csv")
    results = pd.read_csv(base_path + "results.csv")
    drivers = pd.read_csv(base_path + "drivers.csv")
    constructors = pd.read_csv(base_path + "constructors.csv")
    qualifying = pd.read_csv(base_path + "qualifying.csv")
    lap_times = pd.read_csv(base_path + "lap_times.csv")
    status = pd.read_csv(base_path + "status.csv")
    circuits = pd.read_csv(base_path + "circuits.csv")
    seasons = pd.read_csv(base_path + "seasons.csv")
    driver_standings = pd.read_csv(base_path + "driver_standings.csv")
    constructor_standings = pd.read_csv(base_path + "constructor_standings.csv")
    pit_stops = pd.read_csv(base_path + "pit_stops.csv")
    sprint_results = pd.read_csv(base_path + "sprint_results.csv")
    constructor_results = pd.read_csv(base_path + "constructor_results.csv")

    return (
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

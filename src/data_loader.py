import pandas as pd

def load_f1_data(base_path="data/f1/"):
    races = pd.read_csv(base_path + "races.csv")
    results = pd.read_csv(base_path + "results.csv")
    drivers = pd.read_csv(base_path + "drivers.csv")
    constructors = pd.read_csv(base_path + "constructors.csv")
    return races, results, drivers, constructors

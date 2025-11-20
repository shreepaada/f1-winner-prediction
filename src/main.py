from data_loader import load_f1_data
from feature_engineering import build_feature_table, get_features_and_target
from train_models import train_and_evaluate_models

def main():
    races, results, drivers, constructors = load_f1_data()
    df = build_feature_table(races, results, drivers, constructors)

    print(f"Total rows in final training table: {len(df)}")

    X, y, feature_names = get_features_and_target(df)
    train_and_evaluate_models(X, y, feature_names)


if __name__ == "__main__":
    main()

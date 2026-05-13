import json
from pathlib import Path

from sklearn.dummy import DummyRegressor

from src.build_model import (
    build_training_set,
    evaluate,
    feature_importance,
    train_lightgbm,
    train_linear_regression,
)
from src.data import load_player_data


def main():
    df = load_player_data()
    X_train, Y_train, X_test, Y_test = build_training_set(df)

    lr = train_linear_regression(X_train, Y_train)
    lgbm = train_lightgbm(X_train, Y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, Y_train)

    results = [
        evaluate(lr, X_test, Y_test, "LinearRegression"),
        evaluate(lgbm, X_test, Y_test, "LightGBM        "),
        evaluate(baseline, X_test, Y_test, "Baseline (mean) "),
    ]

    # Structured copy of the same numbers. CI parses this to compute deltas.
    Path("metrics.json").write_text(json.dumps(results, indent=2))

    for label, model in [("LinearRegression", lr), ("LightGBM", lgbm)]:
        print(f"\n{label} feature importance (MAE worsening when shuffled):")
        importance = feature_importance(model, X_test, Y_test)
        print(importance.to_string(float_format=lambda v: f"{v:6.2f}"))


if __name__ == "__main__":
    main()

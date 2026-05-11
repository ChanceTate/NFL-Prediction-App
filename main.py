from sklearn.dummy import DummyRegressor

from src.build_model import build_training_set, evaluate, train
from src.data import load_player_data


def main():
    df = load_player_data()
    X_train, Y_train, X_test, Y_test = build_training_set(df)

    model = train(X_train, Y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, Y_train)

    evaluate(model, X_test, Y_test, "LinearRegression")
    evaluate(baseline, X_test, Y_test, "Baseline (mean)")


if __name__ == "__main__":
    main()

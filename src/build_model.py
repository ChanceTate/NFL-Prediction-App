import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import filter_qbs
from src.features import FEATURE_COLS, add_features

TRAIN_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]

TARGET_COL = "passing_yards"


def split_train_test(df: pd.DataFrame):
    """Time-based split: earlier seasons train, later seasons test."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    return (
        train[FEATURE_COLS],
        train[TARGET_COL],
        test[FEATURE_COLS],
        test[TARGET_COL],
    )


def build_training_set(df: pd.DataFrame):
    """Filter to QBs, engineer features, drop unusable rows, split by season."""
    df = filter_qbs(df)
    df = add_features(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return split_train_test(df)


def train(X_train, Y_train) -> LinearRegression:
    # Before we add categorical features, i think we should switch to LGBMClassifier
    # from lightgbm. This way we dont have to deal with encoding stuff
    return LinearRegression().fit(X_train, Y_train)


def evaluate(model, X_test, Y_test, label: str):
    preds = model.predict(X_test)
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)
    print(f"{label} MAE: {mae:.2f}, R²: {r2:.2f}")

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import filter_qbs
from src.features import (
    FEATURE_COLS,
    add_opponent_pass_defense,
    add_rolling_pass_attempts,
    add_rolling_passing_yards,
)

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
    qbs = filter_qbs(df)
    qbs = add_rolling_passing_yards(qbs)
    qbs = add_rolling_pass_attempts(qbs)
    # add_opponent_pass_defense needs the full league-wide df to compute yards
    # allowed across all passers, not just QBs.
    qbs = add_opponent_pass_defense(qbs, df)
    qbs = qbs.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return split_train_test(qbs)


def train_linear_regression(X_train, Y_train) -> LinearRegression:
    return LinearRegression().fit(X_train, Y_train)


def train_lightgbm(X_train, Y_train) -> LGBMRegressor:
    # random_state pinned so CI metrics don't vary from run to run.
    return LGBMRegressor(verbose=-1, random_state=42).fit(X_train, Y_train)


def evaluate(model, X_test, Y_test, label: str):
    preds = model.predict(X_test)
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)
    print(f"{label} MAE: {mae:.2f}, R²: {r2:.2f}")

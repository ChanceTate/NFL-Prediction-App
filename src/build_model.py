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
    # Since we have so few features and so little data at the moment, the default
    # lgbm parameters were causing some overfitting. These are some
    # tweaks i found that helped a bit, but we should keep an eye on this as we add more features
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=30,
        reg_lambda=1.0,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
    ).fit(X_train, Y_train)


def evaluate(model, X_test, Y_test, label: str) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)
    print(f"{label} MAE: {mae:.2f}, R²: {r2:.2f}")
    return {"label": label.strip(), "mae": float(mae), "r2": float(r2)}

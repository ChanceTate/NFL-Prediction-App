import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import filter_qbs
from src.features import (
    FEATURE_COLS,
    add_opponent_pass_defense,
    add_rolling_epa_per_attempt,
    add_rolling_pass_attempts,
    add_rolling_passing_yards,
)

TRAIN_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]

TARGET_COL = "passing_yards"

# Defines which rows enter train/test. New features
# must impute NaN themselves rather than shrink this set, otherwise the
# test rows shift between runs and metrics aren't comparable.
ROW_INCLUSION_FEATURES = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
]


def _assert_no_extra_nans(df: pd.DataFrame, cols: list[str]) -> None:
    """Fails loud if a feature has NaN inside the row universe. That means a
    new feature is silently shrinking the test set. Impute at the feature level."""
    bad = df[cols].isna().sum()
    bad = bad[bad > 0]
    if not bad.empty:
        raise ValueError(
            f"Feature(s) have NaN within row universe: {bad.to_dict()}. "
            "Impute in the feature definition or widen the rolling window."
        )


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
    qbs = add_rolling_epa_per_attempt(qbs)
    qbs = add_opponent_pass_defense(qbs, df)  # full df: includes non-QB passers

    qbs = qbs.dropna(subset=ROW_INCLUSION_FEATURES + [TARGET_COL])
    _assert_no_extra_nans(qbs, FEATURE_COLS)

    return split_train_test(qbs)


def train_linear_regression(X_train, Y_train) -> LinearRegression:
    return LinearRegression().fit(X_train, Y_train)


def train_lightgbm(X_train, Y_train) -> LGBMRegressor:
    # Tuned down from defaults to suppress overfit on small feature set / data.
    # Revisit as features and seasons grow.
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

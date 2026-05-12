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

# The "row universe": features whose missing values define which rows we
# train and test on. Once a feature is in here, the row set is locked. New
# features added to FEATURE_COLS must not introduce new missing values. If
# they do, fix them in the feature code (impute/ widen the window/ etc.).
# Otherwise the test set shifts under us and we can't compare runs.
ROW_INCLUSION_FEATURES = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
]


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
    # add_opponent_pass_defense needs the full league-wide df to compute yards
    # allowed across all passers, not just QBs.
    qbs = add_opponent_pass_defense(qbs, df)
    # Stable row filter: drop only on ROW_INCLUSION_FEATURES + target.
    # Adding more features to FEATURE_COLS no longer changes the row set.
    qbs = qbs.dropna(subset=ROW_INCLUSION_FEATURES + [TARGET_COL])
    # If any FEATURE_COLS still has NaN within the included rows, a recently
    # added feature has a NaN pattern that doesn't match the inclusion
    # universe. Fix the feature (impute or use a wider window) rather than
    # quietly dropping rows here.
    extra_nan = qbs[FEATURE_COLS].isna().sum()
    bad = extra_nan[extra_nan > 0]
    if not bad.empty:
        raise ValueError(
            f"Feature(s) have NaN within row-inclusion universe: {bad.to_dict()}. "
            "Update the feature to impute or use a wider window."
        )
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

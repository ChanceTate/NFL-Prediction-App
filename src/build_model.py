import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.data import filter_qbs
from src.features import (
    FEATURE_COLS,
    add_last_game_vs_season_avg,
    add_opponent_pass_defense,
    add_qb_vs_defense_history,
    add_rolling_epa_per_attempt,
    add_rolling_pass_attempts,
    add_rolling_passing_yards,
    add_rolling_team_plays,
    add_rolling_yds_slope,
    add_top_receiver_rolling,
    add_rolling_passing_air_yards,
    add_rolling_CPOE,
    add_home_away_rolling
)

TARGET_COL = "passing_yards"

# Sliding-window walk-forward CV folds. Each fold trains on exactly 3 seasons
# so training-set size is constant across folds. fold-to-fold variance
# reflects test season difficulty, not "more data helps." We pay for that
# cleaner variance by discarding the oldest training season each fold.
WALK_FORWARD_FOLDS = [
    {"train": [2016, 2017, 2018], "test": [2019]},
    {"train": [2017, 2018, 2019], "test": [2020]},
    {"train": [2018, 2019, 2020], "test": [2021]},
    {"train": [2019, 2020, 2021], "test": [2022]},
    {"train": [2020, 2021, 2022], "test": [2023]},
    {"train": [2021, 2022, 2023], "test": [2024]},
    {"train": [2022, 2023, 2024], "test": [2025]},
]

# Every fold's training seasons must come before its test
# season. Catches the case where we edit the folds and accidentally
# introduce overlap.
for _fold in WALK_FORWARD_FOLDS:
    if max(_fold["train"]) >= min(_fold["test"]):
        raise ValueError(f"Walk-forward fold has train season >= test season: {_fold}")
del _fold

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


def split_train_test(
    df: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
):
    """Time-based split: rows in train_seasons → train; rows in test_seasons → test."""
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return (
        train[FEATURE_COLS],
        train[TARGET_COL],
        test[FEATURE_COLS],
        test[TARGET_COL],
    )


def build_training_set(
    df: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
):
    """Filter to QBs, engineer features, drop unusable rows, split by season."""
    qbs = filter_qbs(df)
    qbs = add_rolling_passing_yards(qbs)
    qbs = add_rolling_pass_attempts(qbs)
    qbs = add_rolling_epa_per_attempt(qbs)
    qbs = add_opponent_pass_defense(qbs, df)  # full df: includes non-QB passers
    qbs = add_rolling_team_plays(qbs, df)  # full df: includes RB/WR carries
    qbs = add_top_receiver_rolling(qbs, df)  # full df: receivers aren't in qbs
    qbs = add_qb_vs_defense_history(qbs)
    qbs = add_rolling_yds_slope(qbs)
    qbs = add_last_game_vs_season_avg(qbs)
    qbs = add_rolling_passing_air_yards(qbs)
    qbs = add_rolling_CPOE(qbs)
    qbs = add_home_away_rolling(qbs)
    qbs = qbs.dropna(subset=ROW_INCLUSION_FEATURES + [TARGET_COL])
    _assert_no_extra_nans(qbs, FEATURE_COLS)

    return split_train_test(qbs, train_seasons, test_seasons)


def train_linear_regression(X_train, Y_train) -> LinearRegression:
    return LinearRegression().fit(X_train, Y_train)


def train_lightgbm(X_train, Y_train) -> LGBMRegressor:
    # Tuned via scripts/tune_lgbm.py.
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=4,
        min_child_samples=10,
        reg_lambda=0.0,
        colsample_bytree=0.5,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
    ).fit(X_train, Y_train)


def feature_importance(model, X_test, Y_test, n_repeats: int = 10) -> pd.Series:
    """Permutation importance: average MAE worsening when each feature is shuffled.
    Higher = more important.
    """
    result = permutation_importance(
        model,
        X_test,
        Y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=42,
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def feature_ablation(train_func, X_train, Y_train, X_test, Y_test) -> pd.Series:
    """Leave-one-out ablation: for each feature, retrain the model without it
    and measure the change in MAE. Returns delta MAE per feature.

    Positive delta = removing the feature hurt the model (feature was contributing
    unique information). Negative delta = removing the feature helped (feature was
    noise or fully redundant with others).
    """
    baseline = train_func(X_train, Y_train)
    baseline_mae = mean_absolute_error(Y_test, baseline.predict(X_test))
    deltas = {}
    for feat in X_train.columns:
        cols = [c for c in X_train.columns if c != feat]
        m = train_func(X_train[cols], Y_train)
        deltas[feat] = mean_absolute_error(Y_test, m.predict(X_test[cols])) - baseline_mae
    return pd.Series(deltas).sort_values(ascending=False)

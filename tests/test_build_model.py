import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.build_model import (
    TARGET_COL,
    TEST_SEASONS,
    TRAIN_SEASONS,
    feature_importance,
    split_train_test,
)
from src.features import FEATURE_COLS


def test_split_train_test_respects_seasons():
    # Include one row per season in TRAIN, TEST, and a season in neither (2019)
    # to verify the split filters by membership in the configured lists.
    seasons = TRAIN_SEASONS + TEST_SEASONS + [2019]
    data = {"season": seasons, TARGET_COL: range(len(seasons))}
    for col in FEATURE_COLS:
        data[col] = range(len(seasons))
    df = pd.DataFrame(data)

    X_train, Y_train, X_test, Y_test = split_train_test(df)

    assert len(X_train) == len(TRAIN_SEASONS)
    assert len(X_test) == len(TEST_SEASONS)
    assert len(Y_train) == len(TRAIN_SEASONS)
    assert len(Y_test) == len(TEST_SEASONS)
    # The 2019 row should be in neither split.
    assert len(X_train) + len(X_test) == len(seasons) - 1


def test_feature_importance_ranks_predictive_feature_highest():
    # Synthetic data: signal column drives the target, noise columns are
    # independent random. A correct importance ranker should put
    # signal on top and the noise features near zero.
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    Y = X["signal"] * 5 + rng.normal(scale=0.1, size=n)

    model = LinearRegression().fit(X, Y)
    importance = feature_importance(model, X, Y, n_repeats=5)

    assert importance.index[0] == "signal"
    assert importance["signal"] > importance["noise_a"]
    assert importance["signal"] > importance["noise_b"]

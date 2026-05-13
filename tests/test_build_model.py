import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.build_model import (
    TARGET_COL,
    WALK_FORWARD_FOLDS,
    feature_importance,
    split_train_test,
)
from src.features import FEATURE_COLS


def test_split_train_test_respects_seasons():
    # Include rows in TRAIN, TEST, and a season in neither (2019) to verify
    # the split filters by membership in the explicit season lists.
    train_seasons = [2020, 2021, 2022]
    test_seasons = [2023]
    seasons = train_seasons + test_seasons + [2019]
    data = {"season": seasons, TARGET_COL: range(len(seasons))}
    for col in FEATURE_COLS:
        data[col] = range(len(seasons))
    df = pd.DataFrame(data)

    X_train, Y_train, X_test, Y_test = split_train_test(df, train_seasons, test_seasons)

    assert len(X_train) == len(train_seasons)
    assert len(X_test) == len(test_seasons)
    assert len(Y_train) == len(train_seasons)
    assert len(Y_test) == len(test_seasons)
    # The 2019 row should be in neither split.
    assert len(X_train) + len(X_test) == len(seasons) - 1


def test_walk_forward_folds_are_time_ordered():
    """Each fold's train seasons must all precede its test season, otherwise
    we'd be 'predicting' the past with the future, leaking information."""
    assert WALK_FORWARD_FOLDS, "expected at least one fold"
    for fold in WALK_FORWARD_FOLDS:
        assert "train" in fold and "test" in fold
        max_train = max(fold["train"])
        min_test = min(fold["test"])
        assert max_train < min_test, (
            f"fold has train seasons {fold['train']} that overlap/follow "
            f"test seasons {fold['test']}"
        )


def test_walk_forward_folds_have_equal_train_size():
    """Sliding window: every fold should train on the same number of seasons,
    so fold-to-fold variance reflects season difficulty, not training size."""
    sizes = {len(fold["train"]) for fold in WALK_FORWARD_FOLDS}
    assert len(sizes) == 1, f"fold train sizes vary: {sizes}"


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

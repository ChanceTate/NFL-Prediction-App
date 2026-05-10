import pandas as pd

from src.build_model import TARGET_COL, TEST_SEASONS, TRAIN_SEASONS, split_train_test
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

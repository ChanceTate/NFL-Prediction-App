import pandas as pd

from src.features import FEATURE_COLS, add_features


def _qb_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "passing_yards": [200, 250, 220, 230, 280],
            "attempts": [25, 30, 28, 27, 32],
        }
    )


def test_add_features_produces_all_declared_features():
    result = add_features(_qb_fixture())

    missing = set(FEATURE_COLS) - set(result.columns)
    assert not missing, f"add_features did not produce declared features: {missing}"


def test_rolling_features_use_only_prior_games():
    result = add_features(_qb_fixture()).sort_values("week").reset_index(drop=True)

    early_weeks = result.loc[result["week"].isin([1, 2, 3]), "rolling_yds_3"]
    assert early_weeks.isna().all(), "Early weeks should be NaN — current game is leaking in"

    expected_w4 = (200 + 250 + 220) / 3
    assert result.loc[result["week"] == 4, "rolling_yds_3"].iloc[0] == expected_w4

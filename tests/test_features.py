import pandas as pd

from src.features import (
    FEATURE_COLS,
    add_opponent_pass_defense,
    add_rolling_pass_attempts,
    add_rolling_passing_yards,
)


def _qb_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "team": ["KC"] * 5,
            "opponent_team": ["DEF"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "passing_yards": [200, 250, 220, 230, 280],
            "attempts": [25, 30, 28, 27, 32],
        }
    )


def _league_fixture() -> pd.DataFrame:
    # The only defense in this fixture ("DEF") allowed exactly the QB's passing
    # yards each week, so the rolling expectation is easy to compute by hand.
    return pd.DataFrame(
        {
            "opponent_team": ["DEF"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "passing_yards": [200, 250, 220, 230, 280],
        }
    )


def test_feature_pipeline_produces_all_declared_features():
    qbs = _qb_fixture()
    qbs = add_rolling_passing_yards(qbs)
    qbs = add_rolling_pass_attempts(qbs)
    qbs = add_opponent_pass_defense(qbs, _league_fixture())

    missing = set(FEATURE_COLS) - set(qbs.columns)
    assert not missing, f"Feature pipeline did not produce declared features: {missing}"


def test_rolling_passing_yards_uses_only_prior_games():
    result = add_rolling_passing_yards(_qb_fixture()).sort_values("week").reset_index(drop=True)

    early = result.loc[result["week"].isin([1, 2, 3]), "rolling_yds_3"]
    assert early.isna().all(), "Early weeks should be NaN — current game is leaking in"

    expected_w4 = (200 + 250 + 220) / 3
    assert result.loc[result["week"] == 4, "rolling_yds_3"].iloc[0] == expected_w4


def test_opponent_pass_defense_uses_only_prior_games():
    result = add_opponent_pass_defense(_qb_fixture(), _league_fixture())
    result = result.sort_values("week").reset_index(drop=True)

    early = result.loc[result["week"].isin([1, 2, 3]), "opp_pass_yds_allowed_3"]
    assert early.isna().all(), "Defense rolling shouldn't see its own current game"

    expected_w4 = (200 + 250 + 220) / 3
    assert result.loc[result["week"] == 4, "opp_pass_yds_allowed_3"].iloc[0] == expected_w4

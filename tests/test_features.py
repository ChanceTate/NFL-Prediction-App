import pandas as pd

from src.features import (
    FEATURE_COLS,
    add_opponent_pass_defense,
    add_rolling_epa_per_attempt,
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
            "attempts": [25, 30, 20, 27, 32],
            "passing_epa": [10.0, 15.0, -5.0, 20.0, 8.0],
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
    qbs = add_rolling_epa_per_attempt(qbs)
    qbs = add_opponent_pass_defense(qbs, _league_fixture())

    missing = set(FEATURE_COLS) - set(qbs.columns)
    assert not missing, f"Feature pipeline did not produce declared features: {missing}"


def test_rolling_passing_yards_uses_only_prior_games():
    result = add_rolling_passing_yards(_qb_fixture()).sort_values("week").reset_index(drop=True)

    early = result.loc[result["week"].isin([1, 2, 3]), "rolling_yds_3"]
    assert early.isna().all(), "Early weeks should be NaN. Current game is leaking in"

    expected_w4 = (200 + 250 + 220) / 3
    assert result.loc[result["week"] == 4, "rolling_yds_3"].iloc[0] == expected_w4


def test_rolling_epa_per_attempt_uses_only_prior_games():
    result = add_rolling_epa_per_attempt(_qb_fixture()).sort_values("week").reset_index(drop=True)

    early = result.loc[result["week"].isin([1, 2, 3]), "rolling_epa_per_att_3"]
    assert early.isna().all(), "Early weeks should be NaN. Current game is leaking in"

    # Volume-weighted: sum(EPA) / sum(attempts) over weeks 1-3
    expected_w4 = (10 + 15 + -5) / (25 + 30 + 20)
    assert result.loc[result["week"] == 4, "rolling_epa_per_att_3"].iloc[0] == expected_w4


def test_rolling_epa_per_attempt_imputes_zero_for_backup_qbs():
    # Backup QB with three prior games of 0 attempts/0 EPA would divide by
    # zero. Should impute to 0 rather than NaN so the row stays in the model.
    backup = pd.DataFrame(
        {
            "player_id": ["backup"] * 4,
            "season": [2023] * 4,
            "week": [1, 2, 3, 4],
            "attempts": [0, 0, 0, 30],
            "passing_epa": [0.0, 0.0, 0.0, 5.0],
        }
    )
    result = add_rolling_epa_per_attempt(backup).sort_values("week").reset_index(drop=True)

    # Week 4 has 3 prior games to roll over, all with 0 attempts should impute
    # to 0 (not NaN) so backup-becoming-starter rows aren't silently dropped.
    assert result.loc[result["week"] == 4, "rolling_epa_per_att_3"].iloc[0] == 0


def test_opponent_pass_defense_within_season_rolling():
    """Week 4+ uses rolling of prior weeks within the season, not including current."""
    result = add_opponent_pass_defense(_qb_fixture(), _league_fixture())
    result = result.sort_values("week").reset_index(drop=True)

    expected_w4 = (200 + 250 + 220) / 3
    assert result.loc[result["week"] == 4, "opp_pass_yds_allowed_3"].iloc[0] == expected_w4


def test_opponent_pass_defense_falls_back_to_league_avg_when_no_prior_season():
    """Earliest season in the dataset has no prior-season baseline, so weeks 1-3
    fall back to the league-wide average rather than producing NaN."""
    result = add_opponent_pass_defense(_qb_fixture(), _league_fixture())
    result = result.sort_values("week").reset_index(drop=True)

    early = result.loc[result["week"].isin([1, 2, 3]), "opp_pass_yds_allowed_3"]
    assert early.notna().all(), "Early weeks should fall back, not be NaN"

    league_avg = (200 + 250 + 220 + 230 + 280) / 5
    assert (early == league_avg).all()


def test_opponent_pass_defense_falls_back_to_prior_season_avg():
    """Early weeks of season N use season N-1's full-season average for that
    defense, not the league-wide average."""
    qbs = pd.DataFrame(
        {
            "player_id": ["p1"] * 2,
            "team": ["KC"] * 2,
            "opponent_team": ["DEF"] * 2,
            "season": [2024, 2024],
            "week": [1, 2],
            "passing_yards": [220, 230],
            "attempts": [28, 27],
            "passing_epa": [5.0, 8.0],
        }
    )
    # 2023: DEF allowed [100, 200, 300, 400, 500] → season avg = 300.
    # 2024: weeks 1-3 should fall back to 300, not the league-wide average.
    league = pd.DataFrame(
        {
            "opponent_team": ["DEF"] * 6,
            "season": [2023, 2023, 2023, 2023, 2023, 2024],
            "week": [1, 2, 3, 4, 5, 1],
            "passing_yards": [100, 200, 300, 400, 500, 220],
        }
    )

    result = add_opponent_pass_defense(qbs, league)

    week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
    assert week1_2024["opp_pass_yds_allowed_3"].iloc[0] == 300

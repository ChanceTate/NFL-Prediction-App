import pandas as pd

from src.features import (
    FEATURE_COLS,
    add_last_game_vs_season_avg,
    add_opponent_pass_defense,
    add_qb_vs_defense_history,
    add_rolling_epa_per_attempt,
    add_rolling_pass_attempts,
    add_rolling_pass_fd_per_att,
    add_rolling_passing_yards,
    add_rolling_team_plays,
    add_rolling_yds_slope,
    add_top_receiver_rolling,
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
            "passing_first_downs": [10, 13, 8, 12, 14],
        }
    )


def _league_fixture() -> pd.DataFrame:
    # The only defense in this fixture ("DEF") allowed exactly the QB's passing
    # yards each week, so the rolling expectation is easy to compute by hand.
    # carries/attempts also let the same fixture serve add_rolling_team_plays.
    return pd.DataFrame(
        {
            "team": ["KC"] * 5,
            "opponent_team": ["DEF"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "passing_yards": [200, 250, 220, 230, 280],
            "attempts": [25, 30, 20, 27, 32],
            "carries": [20, 25, 18, 22, 28],
            "position": ["QB"] * 5,
        }
    )


def _receivers_fixture() -> pd.DataFrame:
    # Two KC receivers so the per-team max picks the higher one.
    return pd.DataFrame(
        {
            "player_id": ["wr1"] * 5 + ["wr2"] * 5,
            "team": ["KC"] * 10,
            "position": ["WR"] * 5 + ["TE"] * 5,
            "season": [2023] * 10,
            "week": [1, 2, 3, 4, 5] * 2,
            "receiving_yards": [80, 100, 60, 90, 110, 50, 70, 40, 60, 80],
        }
    )


def _full_league_fixture() -> pd.DataFrame:
    """League fixture + receivers for features that filter on position."""
    return pd.concat([_league_fixture(), _receivers_fixture()], ignore_index=True)


def test_feature_pipeline_produces_all_declared_features():
    qbs = _qb_fixture()
    qbs = add_rolling_passing_yards(qbs)
    qbs = add_rolling_pass_attempts(qbs)
    qbs = add_rolling_epa_per_attempt(qbs)
    qbs = add_opponent_pass_defense(qbs, _league_fixture())
    qbs = add_rolling_team_plays(qbs, _league_fixture())
    qbs = add_top_receiver_rolling(qbs, _full_league_fixture())
    qbs = add_qb_vs_defense_history(qbs)
    qbs = add_rolling_yds_slope(qbs)
    qbs = add_last_game_vs_season_avg(qbs)
    qbs = add_rolling_pass_fd_per_att(qbs)

    missing = set(FEATURE_COLS) - set(qbs.columns)
    assert not missing, f"Feature pipeline did not produce declared features: {missing}"


def test_rolling_pass_fd_per_att_uses_only_prior_games():
    """Volume-weighted rate of first downs per attempt should use only the
    QB's prior games. Same shift(1).rolling(3) pattern as the other rate
    features."""
    result = add_rolling_pass_fd_per_att(_qb_fixture()).sort_values("week").reset_index(drop=True)

    # Weeks 1-3 have fewer than 3 prior games, so the rolling sum is NaN.
    early = result.loc[result["week"].isin([1, 2, 3]), "rolling_pass_fd_per_att_3"]
    assert early.isna().all(), "Early weeks should be NaN. Current game is leaking in"

    # Week 4: volume-weighted rate over weeks 1-3.
    # sum(fds) = 10 + 13 + 8 = 31; sum(atts) = 25 + 30 + 20 = 75; rate = 31/75
    expected_w4 = (10 + 13 + 8) / (25 + 30 + 20)
    assert result.loc[result["week"] == 4, "rolling_pass_fd_per_att_3"].iloc[0] == expected_w4


def test_rolling_pass_fd_per_att_imputes_zero_for_backup_qbs():
    """Backup QB with three prior games of 0 attempts should impute to 0
    (no signal) rather than NaN, so the row stays in the universe."""
    backup = pd.DataFrame(
        {
            "player_id": ["backup"] * 4,
            "season": [2023] * 4,
            "week": [1, 2, 3, 4],
            "attempts": [0, 0, 0, 25],
            "passing_first_downs": [0, 0, 0, 12],
        }
    )
    result = add_rolling_pass_fd_per_att(backup).sort_values("week").reset_index(drop=True)

    # Week 4 has 3 prior games with 0 attempts → impute 0 instead of NaN
    assert result.loc[result["week"] == 4, "rolling_pass_fd_per_att_3"].iloc[0] == 0


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


def test_rolling_team_plays_within_season_rolling():
    """Week 4+ uses 3-game rolling of prior weeks' team plays (attempts + carries)."""
    result = add_rolling_team_plays(_qb_fixture(), _league_fixture())
    result = result.sort_values("week").reset_index(drop=True)

    # Plays = attempts + carries by week: 45, 55, 38, 49, 60
    expected_w4 = (45 + 55 + 38) / 3
    assert result.loc[result["week"] == 4, "rolling_team_plays_3"].iloc[0] == expected_w4


def test_rolling_team_plays_falls_back_to_prior_season_avg():
    """Early weeks of season N use season N-1's full-season average for that team."""
    qbs = pd.DataFrame(
        {
            "player_id": ["p1"] * 2,
            "team": ["KC"] * 2,
            "opponent_team": ["DEF"] * 2,
            "season": [2024, 2024],
            "week": [1, 2],
        }
    )
    league = pd.DataFrame(
        {
            "team": ["KC"] * 6,
            "season": [2023, 2023, 2023, 2023, 2023, 2024],
            "week": [1, 2, 3, 4, 5, 1],
            "attempts": [30, 35, 40, 25, 50, 30],
            "carries": [20, 25, 30, 15, 30, 25],
        }
    )
    result = add_rolling_team_plays(qbs, league)
    week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
    assert week1_2024["rolling_team_plays_3"].iloc[0] == 60


def test_top_receiver_rolling_picks_max_among_team_receivers():
    """Week 4 should reflect WR1's rolling (higher yards), not WR2's."""
    result = add_top_receiver_rolling(_qb_fixture(), _full_league_fixture())
    result = result.sort_values("week").reset_index(drop=True)

    # WR1 rolling weeks 1-3: (80 + 100 + 60) / 3 = 80
    # WR2 rolling weeks 1-3: (50 + 70 + 40) / 3 = 53.33...
    # Max = 80 (WR1)
    expected_w4 = (80 + 100 + 60) / 3
    assert result.loc[result["week"] == 4, "top_receiver_rolling_yds_3"].iloc[0] == expected_w4


def test_top_receiver_rolling_falls_back_to_prior_season_avg():
    """Early weeks of season N use season N-1 team avg of top receiver rolling."""
    qbs = pd.DataFrame(
        {
            "player_id": ["p1"] * 2,
            "team": ["KC"] * 2,
            "opponent_team": ["DEF"] * 2,
            "season": [2024, 2024],
            "week": [1, 2],
        }
    )
    # WR1 played all of 2023 (avg 100 yds/game). Brand-new WR in 2024 week 1
    # so current rolling is NaN. This forces the prior-season fallback.
    league = pd.DataFrame(
        {
            "player_id": ["wr1"] * 5 + ["wr_new"],
            "team": ["KC"] * 6,
            "position": ["WR"] * 6,
            "season": [2023, 2023, 2023, 2023, 2023, 2024],
            "week": [1, 2, 3, 4, 5, 1],
            "receiving_yards": [100, 100, 100, 100, 100, 50],
        }
    )
    result = add_top_receiver_rolling(qbs, league)

    # 2023 KC weekly top-rolling: NaN, NaN, NaN, 100, 100 → mean = 100.
    # 2024 week 1 has only wr_new with no history → current rolling NaN
    # → falls back to prior-season avg = 100.
    week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
    assert week1_2024["top_receiver_rolling_yds_3"].iloc[0] == 100


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


def test_qb_vs_defense_history_uses_only_prior_matchups():
    """Repeated matchups against the same defense should give expanding mean
    of prior games only. Current game must not leak in."""
    # Mahomes vs DEF in weeks 1, 3, 5 (each with different yards). Weeks 2 and
    # 4 are non-DEF games to make sure they don't pollute the matchup average.
    qbs = pd.DataFrame(
        {
            "player_id": ["mahomes"] * 5,
            "opponent_team": ["DEF", "OTHER", "DEF", "OTHER", "DEF"],
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "passing_yards": [200, 250, 220, 230, 280],
        }
    )
    result = add_qb_vs_defense_history(qbs).sort_values("week").reset_index(drop=True)

    # Week 3 vs DEF: only prior DEF matchup is week 1 (200). Mean of [200] = 200.
    w3 = result.loc[result["week"] == 3, "qb_vs_def_avg_yds"].iloc[0]
    assert w3 == 200

    # Week 5 vs DEF: prior DEF matchups are weeks 1 and 3 (200, 220). Mean = 210.
    w5 = result.loc[result["week"] == 5, "qb_vs_def_avg_yds"].iloc[0]
    assert w5 == 210


def test_qb_vs_defense_history_falls_back_to_career_avg_for_new_defense():
    """When the QB has career history but has never faced this specific defense,
    fall back to QB's career-to-date passing yards average."""
    qbs = pd.DataFrame(
        {
            "player_id": ["mahomes"] * 4,
            "opponent_team": ["DEF_A", "DEF_A", "DEF_A", "DEF_B"],  # Week 4 = new defense
            "season": [2023] * 4,
            "week": [1, 2, 3, 4],
            "passing_yards": [200, 250, 220, 230],
        }
    )
    result = add_qb_vs_defense_history(qbs).sort_values("week").reset_index(drop=True)

    # Week 4 is Mahomes' first game vs DEF_B, so matchup_avg is NaN. Fallback
    # is his career-to-date avg: mean of weeks 1, 2, 3 = (200+250+220)/3 = 223.33...
    expected = (200 + 250 + 220) / 3
    w4 = result.loc[result["week"] == 4, "qb_vs_def_avg_yds"].iloc[0]
    assert w4 == expected


def test_qb_vs_defense_history_falls_back_to_league_avg_for_first_game():
    """A QB's very first NFL game has no matchup history AND no career history.
    Should fall back to the league-wide average (mean of all passing_yards in df)."""
    qbs = pd.DataFrame(
        {
            "player_id": ["rookie", "vet", "vet"],
            "opponent_team": ["DEF_A", "DEF_B", "DEF_C"],
            "season": [2023] * 3,
            "week": [1, 1, 2],
            "passing_yards": [180, 250, 270],
        }
    )
    result = add_qb_vs_defense_history(qbs)

    # Rookie's only game has no matchup history and no career history. Falls
    # all the way to league avg = mean of [180, 250, 270] = 233.33...
    league_avg = (180 + 250 + 270) / 3
    rookie_w1 = result[(result["player_id"] == "rookie") & (result["week"] == 1)]
    assert rookie_w1["qb_vs_def_avg_yds"].iloc[0] == league_avg


def test_rolling_yds_slope_uses_only_prior_games():
    """Slope should use the QB's prior games only (no current-game leakage). For
    3 evenly-spaced points the OLS slope equals (last - first) / 2."""
    result = add_rolling_yds_slope(_qb_fixture()).sort_values("week").reset_index(drop=True)

    # Weeks 1-3: shift(3) is NaN, so slope is NaN. Same NaN pattern as rolling_yds_3.
    early = result.loc[result["week"].isin([1, 2, 3]), "rolling_yds_slope_3"]
    assert early.isna().all(), "Early weeks should be NaN. Current game is leaking in"

    # Week 4: slope from games 1, 2, 3 is (yards_w3 - yards_w1) / 2 = (220 - 200) / 2 = 10.
    expected_w4 = (220 - 200) / 2
    assert result.loc[result["week"] == 4, "rolling_yds_slope_3"].iloc[0] == expected_w4

    # Week 5: slope from games 2, 3, 4 is (yards_w4 - yards_w2) / 2 = (230 - 250) / 2 = -10.
    expected_w5 = (230 - 250) / 2
    assert result.loc[result["week"] == 5, "rolling_yds_slope_3"].iloc[0] == expected_w5


def test_last_game_vs_season_avg_uses_only_prior_games():
    """Gap feature: previous game's yards minus QB's season-to-date avg.
    Both pieces must use shift(1) so the current game doesn't leak in."""
    result = add_last_game_vs_season_avg(_qb_fixture()).sort_values("week").reset_index(drop=True)

    # Week 1: no prior game, season avg also NaN. Falls back to 0 by design.
    w1 = result.loc[result["week"] == 1, "last_game_vs_season_avg"].iloc[0]
    assert w1 == 0

    # Week 2: last game = week 1 (200), season avg = mean of [200] = 200, diff = 0.
    w2 = result.loc[result["week"] == 2, "last_game_vs_season_avg"].iloc[0]
    assert w2 == 0

    # Week 3: last game = week 2 (250), season avg = mean([200, 250]) = 225, diff = +25.
    w3 = result.loc[result["week"] == 3, "last_game_vs_season_avg"].iloc[0]
    assert w3 == 250 - 225

    # Week 4: last game = week 3 (220), season avg = mean([200, 250, 220]) = 223.33,
    # diff = 220 - 223.33 = -3.33.
    season_avg_at_w4 = (200 + 250 + 220) / 3
    w4 = result.loc[result["week"] == 4, "last_game_vs_season_avg"].iloc[0]
    assert w4 == 220 - season_avg_at_w4

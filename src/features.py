import pandas as pd

from src import data

# Schedules carry historical codes (SD/OAK) for pre-relocation seasons while
# player_stats uses the current codes. Normalize before joining.
_SCHEDULE_TEAM_REMAP = {"SD": "LAC", "OAK": "LV"}

FEATURE_COLS = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
    "rolling_epa_per_att_3",
    "rolling_team_plays_3",
    "top_receiver_rolling_yds_3",
    "qb_vs_def_avg_yds",
    "rolling_yds_slope_3",
    "last_game_vs_season_avg",
    "rolling_pass_fd_per_att_3",
    "rolling_team_points_3",
]

# Positions that catch passes. Excludes defenders (who have 0 receiving_yards
# and would just waste compute) and QBs (rarely catch passes, would clutter
# the per-team max).
RECEIVING_POSITIONS = {"WR", "TE", "RB", "FB"}


def add_rolling_passing_yards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    df["rolling_yds_3"] = df.groupby("player_id")["passing_yards"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    return df


def add_rolling_pass_attempts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    df["rolling_pass_atts_3"] = df.groupby("player_id")["attempts"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    return df


def add_rolling_pass_fd_per_att(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    # passing_first_downs is the count of pass plays that earned a first down.
    # Fill NaN as 0 in case it's missing for any rows
    fds = df["passing_first_downs"].fillna(0)

    # Volume-weighted: sum(first_downs) / sum(attempts) over the prior 3 games.
    rolling_fds = fds.groupby(df["player_id"]).transform(lambda x: x.shift(1).rolling(3).sum())
    rolling_atts = df.groupby("player_id")["attempts"].transform(
        lambda x: x.shift(1).rolling(3).sum()
    )
    ratio = rolling_fds / rolling_atts.where(rolling_atts != 0)
    # Backups with 0 attempts across all 3 prior games: impute 0 (no signal) so
    # the row stays in the universe instead of being dropped.
    no_signal = rolling_atts.notna() & (rolling_atts == 0)
    df["rolling_pass_fd_per_att_3"] = ratio.mask(no_signal, 0)
    return df


def add_rolling_epa_per_attempt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    # nflreadpy leaves passing_epa = NaN when a player had 0 attempts (didn't
    # throw at all). Treat those as 0 EPA.
    epa = df["passing_epa"].fillna(0)

    # Volume-weighted ratio: sum(EPA) / sum(attempts) over the prior 3 games.
    rolling_epa = epa.groupby(df["player_id"]).transform(lambda x: x.shift(1).rolling(3).sum())
    rolling_atts = df.groupby("player_id")["attempts"].transform(
        lambda x: x.shift(1).rolling(3).sum()
    )
    ratio = rolling_epa / rolling_atts.where(rolling_atts != 0)
    # Backups with 0 attempts across all 3 prior games: impute 0 (matches their
    # rolling_yds_3 = 0). Preserves true early-season NaN where history is missing.
    no_signal = rolling_atts.notna() & (rolling_atts == 0)
    df["rolling_epa_per_att_3"] = ratio.mask(no_signal, 0)
    return df


def add_opponent_pass_defense(qb_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    # Yards allowed by a defense in (season, week) = sum of passing_yards by every
    # player whose opponent_team was that defense. full_df (not qbs only) so
    # trick-play / RB passes also count toward the defense's allowed total.
    allowed = (
        full_df.groupby(["opponent_team", "season", "week"], as_index=False)["passing_yards"]
        .sum()
        .rename(columns={"opponent_team": "def_team", "passing_yards": "pass_yds_allowed"})
    )
    allowed = allowed.sort_values(["def_team", "season", "week"]).reset_index(drop=True)

    # Rolling within a season. Resets at season boundary so the feature reflects
    # current-season form
    allowed["rolling"] = allowed.groupby(["def_team", "season"])["pass_yds_allowed"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Early-season fallback: each season's first weeks use last year's full-season
    # average for that defense. Built by averaging per (def, season) and rekeying
    # to season+1 so the merge lands on the following year.
    prior_season = (
        allowed.groupby(["def_team", "season"], as_index=False)["pass_yds_allowed"]
        .mean()
        .rename(columns={"pass_yds_allowed": "prior_season_avg"})
    )
    prior_season["season"] = prior_season["season"] + 1
    allowed = allowed.merge(prior_season, on=["def_team", "season"], how="left")

    # Final fallback: league-wide average. Only kicks in for the earliest season
    # in the dataset, where no prior-season baseline exists.
    league_avg = allowed["pass_yds_allowed"].mean()

    allowed["opp_pass_yds_allowed_3"] = (
        allowed["rolling"].fillna(allowed["prior_season_avg"]).fillna(league_avg)
    )

    return qb_df.merge(
        allowed[["def_team", "season", "week", "opp_pass_yds_allowed_3"]],
        left_on=["opponent_team", "season", "week"],
        right_on=["def_team", "season", "week"],
        how="left",
    ).drop(columns=["def_team"])


def add_rolling_team_plays(qb_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    # Total team offensive plays per game = sum of pass attempts + rush carries
    # across all players on that team in (season, week). Computed from full_df
    # so RB and WR carries also count toward team pace.
    plays = full_df.groupby(["team", "season", "week"], as_index=False).agg(
        attempts=("attempts", "sum"),
        carries=("carries", "sum"),
    )
    plays["total_plays"] = plays["attempts"] + plays["carries"]
    plays = plays.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Rolling within season. Resets at season boundary so the feature reflects
    # current-season pace
    plays["rolling"] = plays.groupby(["team", "season"])["total_plays"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Early-season fallback: weeks 1-3 of season N use season N-1's full-season
    # average for that team. Built by averaging per (team, season) and rekeying
    # to season+1 so the merge lands on the following year.
    prior_season = (
        plays.groupby(["team", "season"], as_index=False)["total_plays"]
        .mean()
        .rename(columns={"total_plays": "prior_season_avg"})
    )
    prior_season["season"] = prior_season["season"] + 1
    plays = plays.merge(prior_season, on=["team", "season"], how="left")

    # Final fallback: league-wide average. Only kicks in for the earliest season,
    # where no prior-season baseline exists.
    league_avg = plays["total_plays"].mean()

    plays["rolling_team_plays_3"] = (
        plays["rolling"].fillna(plays["prior_season_avg"]).fillna(league_avg)
    )

    return qb_df.merge(
        plays[["team", "season", "week", "rolling_team_plays_3"]],
        on=["team", "season", "week"],
        how="left",
    )


def add_rolling_team_points(qb_df: pd.DataFrame) -> pd.DataFrame:
    # Game-script proxy. Teams that have been losing recently throw more in the
    # next game (they're trailing more often). Built from schedules' home_score
    # and away_score columns, stacked so each (season, week, team) maps to that
    # team's points in that game.
    schedules = data.load_schedules()
    home = schedules[["season", "week", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points"}
    )
    away = schedules[["season", "week", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points"}
    )
    team_pts = pd.concat([home, away], ignore_index=True)
    team_pts["team"] = team_pts["team"].replace(_SCHEDULE_TEAM_REMAP)
    team_pts = team_pts.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Rolling within season so the feature reflects current-season scoring form.
    team_pts["rolling"] = team_pts.groupby(["team", "season"])["points"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Same fallback chain as other team features: current → prior-season team
    # avg → league avg.
    prior_season = (
        team_pts.groupby(["team", "season"], as_index=False)["points"]
        .mean()
        .rename(columns={"points": "prior_season_avg"})
    )
    prior_season["season"] = prior_season["season"] + 1
    team_pts = team_pts.merge(prior_season, on=["team", "season"], how="left")

    league_avg = team_pts["points"].mean()

    team_pts["rolling_team_points_3"] = (
        team_pts["rolling"].fillna(team_pts["prior_season_avg"]).fillna(league_avg)
    )

    return qb_df.merge(
        team_pts[["team", "season", "week", "rolling_team_points_3"]],
        on=["team", "season", "week"],
        how="left",
    )


def add_top_receiver_rolling(qb_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    # For each (team, season, week), compute the max rolling 3-game receiving
    # yards across the team's receiving-position players.
    receivers = (
        full_df[full_df["position"].isin(RECEIVING_POSITIONS)]
        .sort_values(["player_id", "season", "week"])
        .copy()
    )
    receivers["rec_rolling"] = receivers.groupby("player_id")["receiving_yards"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Max across the team's receivers per (team, season, week). NaN values in
    # rec_rolling (early-season for that receiver) are skipped by .max().
    # If any one receiver on the team has history, the team gets a value.
    team_top = receivers.groupby(["team", "season", "week"], as_index=False)["rec_rolling"].max()
    team_top = team_top.rename(columns={"rec_rolling": "rolling"})
    team_top = team_top.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Early-season fallback: prior-season average top-receiver-rolling for the team.
    prior_season = (
        team_top.groupby(["team", "season"], as_index=False)["rolling"]
        .mean()
        .rename(columns={"rolling": "prior_season_avg"})
    )
    prior_season["season"] = prior_season["season"] + 1
    team_top = team_top.merge(prior_season, on=["team", "season"], how="left")

    # League-wide fallback for the earliest season
    league_avg = team_top["rolling"].mean()

    team_top["top_receiver_rolling_yds_3"] = (
        team_top["rolling"].fillna(team_top["prior_season_avg"]).fillna(league_avg)
    )

    return qb_df.merge(
        team_top[["team", "season", "week", "top_receiver_rolling_yds_3"]],
        on=["team", "season", "week"],
        how="left",
    )


def add_qb_vs_defense_history(df: pd.DataFrame) -> pd.DataFrame:
    # Sorting by player + chronology means each (player_id, opponent_team)
    # subgroup is also chronologically ordered, which is what shift+expanding
    # below relies on.
    df = df.sort_values(["player_id", "season", "week"]).copy()

    # Matchup-specific: this QB's mean passing yards in prior games against
    # this defense. shift(1) excludes the current game; expanding takes mean
    # of all prior matchups regardless of how many.
    matchup_avg = df.groupby(["player_id", "opponent_team"])["passing_yards"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Fallback 1: QB's overall career-to-date average. Used when this QB has
    # never faced this specific defense before, but does have other history.
    career_avg = df.groupby("player_id")["passing_yards"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Fallback 2: league-wide average. Catches the QB's first ever NFL game,
    # where neither matchup nor career history exists.
    league_avg = df["passing_yards"].mean()

    df["qb_vs_def_avg_yds"] = matchup_avg.fillna(career_avg).fillna(league_avg)
    return df


def add_rolling_yds_slope(df: pd.DataFrame) -> pd.DataFrame:
    # Direction of the QB's recent passing yards trend. For 3 evenly-spaced
    # points the OLS slope simplifies to (last - first) / 2, so no fitting
    # needed.
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id")["passing_yards"]
    df["rolling_yds_slope_3"] = (grouped.shift(1) - grouped.shift(3)) / 2
    return df


def add_last_game_vs_season_avg(df: pd.DataFrame) -> pd.DataFrame:
    # Last game's passing yards minus the QB's season-to-date avg.
    df = df.sort_values(["player_id", "season", "week"]).copy()
    last_game = df.groupby("player_id")["passing_yards"].shift(1)
    season_avg = df.groupby(["player_id", "season"])["passing_yards"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["last_game_vs_season_avg"] = (last_game - season_avg).fillna(0)
    return df

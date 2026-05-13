import pandas as pd

FEATURE_COLS = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
    "rolling_epa_per_att_3",
    "rolling_team_plays_3",
]


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

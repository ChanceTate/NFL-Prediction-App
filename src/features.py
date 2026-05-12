import pandas as pd

FEATURE_COLS = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
    "rolling_epa_per_att_3",
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
    # throw at all). Treat those as 0 EPA — semantically correct since "no
    # passes attempted" means "no passing EPA" — and necessary to keep the
    # rolling sum from propagating NaN through the next 3 games.
    epa = df["passing_epa"].fillna(0)

    # Volume-weighted ratio: sum(EPA) / sum(attempts) over the prior 3 games.
    # This is how multi-game EPA-per-attempt is normally reported, and it
    # weights big-volume games more heavily (which is sensible — a 50-attempt
    # game tells you more about the QB than a 5-attempt cleanup appearance).
    rolling_epa = epa.groupby(df["player_id"]).transform(lambda x: x.shift(1).rolling(3).sum())
    rolling_atts = df.groupby("player_id")["attempts"].transform(
        lambda x: x.shift(1).rolling(3).sum()
    )
    ratio = rolling_epa / rolling_atts.where(rolling_atts != 0)
    # Backup QBs may have 0 attempts in every game of the rolling window — the
    # ratio is then 0/0 = NaN, but we want to keep these rows in the model
    # (a backup making their first start is a legitimate prediction target).
    # Impute 0, consistent with the rolling_yds_3 = 0 and rolling_pass_atts_3
    # = 0 these same rows already have. Early-season NaN (insufficient history)
    # is preserved by checking rolling_atts.notna().
    no_signal = rolling_atts.notna() & (rolling_atts == 0)
    df["rolling_epa_per_att_3"] = ratio.mask(no_signal, 0)
    return df


def add_opponent_pass_defense(qb_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    # Passing yards allowed by a defense in a game = sum of passing_yards by
    # every player whose opponent_team was that defense. Computed from the full
    # league-wide df (not just QBs) so trick-play / RB passes still count.
    # not sure if that's how it should be, but seems more accurate than only counting yards from QBs
    allowed = (
        full_df.groupby(["opponent_team", "season", "week"], as_index=False)["passing_yards"]
        .sum()
        .rename(columns={"opponent_team": "def_team", "passing_yards": "pass_yds_allowed"})
    )
    allowed = allowed.sort_values(["def_team", "season", "week"])

    # shift(1) so a defense's current-game yards allowed never leak into its own row
    allowed["opp_pass_yds_allowed_3"] = allowed.groupby("def_team")["pass_yds_allowed"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Merge onto QB rows: each QB faces opponent_team in (season, week), which maps
    # to that defense's def_team key in the allowed table.
    return qb_df.merge(
        allowed[["def_team", "season", "week", "opp_pass_yds_allowed_3"]],
        left_on=["opponent_team", "season", "week"],
        right_on=["def_team", "season", "week"],
        how="left",
    ).drop(columns=["def_team"])

import pandas as pd

FEATURE_COLS = [
    "rolling_yds_3",
    "rolling_pass_atts_3",
    "opp_pass_yds_allowed_3",
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

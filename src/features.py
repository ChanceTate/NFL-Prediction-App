import pandas as pd

FEATURE_COLS = ["rolling_yds_3", "rolling_pass_atts_3", "home_away_3"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id")
    df["rolling_yds_3"] = grouped["passing_yards"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["rolling_pass_atts_3"] = grouped["attempts"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    # I think the opponents allowed passing yards would be really useful. There's not a
    # column for this, but we could calculate it by doing some fancy groupings

    # I think home/away could be useful
    #UNTESTED
    df["home_away_3"] = grouped["home_away"].transform(lambda x: x.shift(1).rolling(3).mean())
    return df
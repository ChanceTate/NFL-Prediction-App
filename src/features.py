import pandas as pd

FEATURE_COLS = ["rolling_yds_3", "rolling_pass_atts_3"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id")
    df["rolling_yds_3"] = grouped["passing_yards"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df["rolling_pass_atts_3"] = grouped["attempts"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    return df

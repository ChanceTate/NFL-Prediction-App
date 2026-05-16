from pathlib import Path

import nflreadpy as nfl
import pandas as pd

SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
# Cache filename includes the season range so changing SEASONS auto-busts the
# cache instead of silently serving stale data.
CACHE = Path(f"data/player_stats_{min(SEASONS)}-{max(SEASONS)}.parquet")


def load_player_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = nfl.load_player_stats(SEASONS).to_pandas()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def filter_qbs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["position"] == "QB"]

#function to add home and away teams, returns home as 0 and away as 1
def home_away(df: pd.DataFrame) -> pd.DataFrame:
    schedules = nfl.load_schedules(SEASONS).to_pandas()
    df = df.merge(schedules[['game_id', 'home_team']], on='game_id', how='left')

    df['is_home'] = (df['team'] == df['home_team']).astype(int)
    df = df.drop(columns='home_team')

    return df
"""
def vegas_lines(df: pd.DataFrame) -> pd.DataFrame:
    pbp = nfl.load_pbp(SEASONS).to_pandas()

    vegas = (
        pbp[["game_id", "spread_line", "total_line"]].drop_duplicates(subset="game_id")
    )
    df = df.merge(vegas, on="game_id", how="left")
    df["spread_line_adjusted"]  = df.apply(
        lambda x: x["spread_line"] if x["is_home"] == 1 else -x["spread_line"], axis=1
    )
    df["implied_team_total"] = (df["total_line"] / 2) + (df["spread_line_adjusted"] / 2)

    return df
"""
"""
df = nfl.load_player_stats(SEASONS).to_pandas()
pbp = nfl.load_pbp(SEASONS).to_pandas()
print(df["game_id"].iloc[0])
print(pbp["game_id"].iloc[0])
"""
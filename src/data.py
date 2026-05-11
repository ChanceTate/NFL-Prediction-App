from pathlib import Path

import nflreadpy as nfl
import pandas as pd

CACHE = Path("data/player_stats.parquet")


def load_player_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = nfl.load_player_stats([2020, 2021, 2022, 2023, 2024, 2025]).to_pandas()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def filter_qbs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["position"] == "QB"]

#function to add home and away teams, returns home as 0 and away as 1
def home_away(df: pd.DataFrame) -> pd.DataFrame:
    schedules = nfl.load_schedules([2020, 2021, 2022, 2023, 2024, 2025]).to_pandas()
    df = df.merge(schedules[['game_id', 'home_team']], on='game_id', how='left')

    df['home_away'] = (df['team'] != df['home_team']).astype(int)

    return df


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

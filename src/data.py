import pandas as pd
import nflreadpy as nfl
from pathlib import Path


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


if __name__ == "__main__":
    df = load_player_data()
    pd.set_option("display.max_columns", None)
    print(list(df.columns))

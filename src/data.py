from pathlib import Path

import nflreadpy as nfl
import pandas as pd

SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
# Cache filename includes the season range so changing SEASONS auto-busts the
# cache instead of silently serving stale data.
CACHE = Path(f"data/player_stats_{min(SEASONS)}-{max(SEASONS)}.parquet")
SCHEDULES_CACHE = Path(f"data/schedules_{min(SEASONS)}-{max(SEASONS)}.parquet")


def load_player_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = nfl.load_player_stats(SEASONS).to_pandas()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def load_schedules() -> pd.DataFrame:
    if SCHEDULES_CACHE.exists():
        return pd.read_parquet(SCHEDULES_CACHE)
    df = nfl.load_schedules(SEASONS).to_pandas()
    SCHEDULES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCHEDULES_CACHE)
    return df


def filter_qbs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["position"] == "QB"]


# Team relocations: schedules carry the historical code, player_stats uses the
# current one. Normalize schedules to match player_stats before joining.
_TEAM_CODE_REMAP = {"SD": "LAC", "OAK": "LV"}


def home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Attach an is_home (1/0) column to df by joining schedules on
    (season, week, team). We avoid joining on game_id because nflreadpy's
    player_stats has missing/inconsistent game_id values; (season, week, team)
    is the canonical key — each team plays at most one game per week.
    """
    schedules = load_schedules().copy()
    schedules["home_team"] = schedules["home_team"].replace(_TEAM_CODE_REMAP)
    schedules["away_team"] = schedules["away_team"].replace(_TEAM_CODE_REMAP)

    # Schedules has one row per game with both home_team and away_team. Stack
    # to long form so each (season, week, team) maps cleanly to its is_home flag.
    home_rows = schedules[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    home_rows["is_home"] = 1
    away_rows = schedules[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    away_rows["is_home"] = 0
    long = pd.concat([home_rows, away_rows], ignore_index=True)

    return df.merge(long, on=["season", "week", "team"], how="left")

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
TEAM_CODE_REMAP = {"SD": "LAC", "OAK": "LV"}


def attach_schedule_columns(
    df: pd.DataFrame, schedules: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Attach schedule columns to df keyed on (season, week, team). cols must
    be values shared by both teams in a game (roof, wind, surface, gameday,
    referee, etc.).

    Joining on (season, week, team) rather than game_id because player_stats'
    game_ids are missing/inconsistent; each team plays at most one game per week.
    """
    s = schedules.copy()
    s["home_team"] = s["home_team"].replace(TEAM_CODE_REMAP)
    s["away_team"] = s["away_team"].replace(TEAM_CODE_REMAP)

    home = s[["season", "week", "home_team", *cols]].rename(columns={"home_team": "team"})
    away = s[["season", "week", "away_team", *cols]].rename(columns={"away_team": "team"})
    long = pd.concat([home, away], ignore_index=True)
    return df.merge(long, on=["season", "week", "team"], how="left")


def home_away(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach an is_home (1/0) column to df. Per-side, so it doesn't fit the
    attach_schedule_columns shape — needs its own stack."""
    s = schedules.copy()
    s["home_team"] = s["home_team"].replace(TEAM_CODE_REMAP)
    s["away_team"] = s["away_team"].replace(TEAM_CODE_REMAP)

    home = s[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    home["is_home"] = 1
    away = s[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    away["is_home"] = 0
    long = pd.concat([home, away], ignore_index=True)
    return df.merge(long, on=["season", "week", "team"], how="left")

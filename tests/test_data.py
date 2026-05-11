import pandas as pd

from src.data import filter_qbs


def test_filter_qbs_keeps_only_qbs():
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "position": ["QB", "WR", "RB", "QB"],
            "passing_yards": [250, 0, 0, 180],
        }
    )

    result = filter_qbs(df)

    assert list(result["position"].unique()) == ["QB"]
    assert list(result["player_id"]) == ["p1", "p4"]

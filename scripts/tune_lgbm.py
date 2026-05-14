"""Exploratory LightGBM hyperparameter tuning.

Runs walk-forward CV for many configurations and
reports mean MAE per config. Two flavors of search:

1. One-at-a-time sweeps: hold all params at the current config, vary one at
   a time. Easy to read what each parameter does in isolation.
2. Themed combinations: explore corners of the space the sweeps can't reach.

Run with `uv run python scripts/tune_lgbm.py`
and read the output, then update train_lightgbm in build_model.py if a config
clearly wins.
"""

import sys
import time
from pathlib import Path

# Make project root importable when run directly: `uv run python scripts/tune_lgbm.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

from src.build_model import WALK_FORWARD_FOLDS, build_training_set  # noqa: E402
from src.data import load_player_data, load_schedules  # noqa: E402

CURRENT = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=30,
    reg_lambda=1.0,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

SWEEPS = {
    "num_leaves": [4, 8, 12, 16, 24, 31],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [10, 20, 30, 50, 100],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "subsample": [0.5, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "n_estimators": [100, 200, 400, 800],
}

# Themed combinations to explore corners of the space single-param sweeps miss.
THEMES = {
    "more_trees_slow_lr": {"n_estimators": 800, "learning_rate": 0.01},
    "more_trees_med_lr": {"n_estimators": 400, "learning_rate": 0.02},
    "bigger_trees_heavy_reg": {
        "num_leaves": 16,
        "reg_lambda": 3.0,
        "min_child_samples": 20,
    },
    "small_trees_no_reg": {"num_leaves": 4, "reg_lambda": 0.0, "min_child_samples": 10},
    "balanced_capacity": {
        "num_leaves": 12,
        "learning_rate": 0.02,
        "n_estimators": 400,
        "min_child_samples": 20,
    },
    "heavy_subsample": {"subsample": 0.5, "colsample_bytree": 0.5},
    "minimal_reg": {"reg_lambda": 0.0, "reg_alpha": 0.0, "min_child_samples": 10},
    "max_reg": {"reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_samples": 50},
}


def evaluate_config(folds_data, params):
    maes = []
    for X_train, Y_train, X_test, Y_test in folds_data:
        model = LGBMRegressor(**params).fit(X_train, Y_train)
        pred = model.predict(X_test)
        maes.append(mean_absolute_error(Y_test, pred))
    return float(np.mean(maes)), float(np.std(maes, ddof=1))


def main():
    df = load_player_data()
    schedules = load_schedules()
    # Pre-build folds once. build_training_set is a meaningful share of the
    # per-config cost; doing it once here saves significant runtime.
    print("Building folds...")
    folds_data = []
    for fold in WALK_FORWARD_FOLDS:
        folds_data.append(
            build_training_set(
                df, schedules, train_seasons=fold["train"], test_seasons=fold["test"]
            )
        )
    print(f"Built {len(folds_data)} folds.\n")

    start = time.time()
    baseline_mean, baseline_std = evaluate_config(folds_data, CURRENT)
    print(f"Baseline (current config): MAE = {baseline_mean:.3f} +/- {baseline_std:.3f}")
    print(f"  ({time.time() - start:.1f}s)\n")

    results = []

    for param, values in SWEEPS.items():
        print(f"=== Sweeping {param} (others held at current) ===")
        for v in values:
            cfg = {**CURRENT, param: v}
            t = time.time()
            mean, std = evaluate_config(folds_data, cfg)
            delta = mean - baseline_mean
            marker = " *" if delta < -0.05 else ("  " if abs(delta) <= 0.05 else "  ")
            print(
                f"  {param}={v!s:<8} MAE = {mean:.3f} +/- {std:.3f}  "
                f"(delta={delta:+.3f}){marker}  [{time.time() - t:.1f}s]"
            )
            results.append({"label": f"{param}={v}", "mae": mean, "std": std, "delta": delta})
        print()

    print("=== Themed multi-param combinations ===")
    for name, overrides in THEMES.items():
        cfg = {**CURRENT, **overrides}
        t = time.time()
        mean, std = evaluate_config(folds_data, cfg)
        delta = mean - baseline_mean
        marker = " *" if delta < -0.05 else "  "
        print(
            f"  {name:<30} MAE = {mean:.3f} +/- {std:.3f}  "
            f"(delta={delta:+.3f}){marker}  [{time.time() - t:.1f}s]"
        )
        results.append({"label": f"theme:{name}", "mae": mean, "std": std, "delta": delta})
    print()

    # Top performers
    results_sorted = sorted(results, key=lambda r: r["mae"])
    print("=== Top 10 configs by mean MAE ===")
    for r in results_sorted[:10]:
        print(
            f"  {r['label']:<35} MAE = {r['mae']:.3f} +/- {r['std']:.3f}  (delta={r['delta']:+.3f})"
        )

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

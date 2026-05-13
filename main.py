import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.build_model import (
    WALK_FORWARD_FOLDS,
    build_training_set,
    feature_ablation,
    feature_importance,
    train_lightgbm,
    train_linear_regression,
)
from src.data import load_player_data

# Models tracked across folds.
TRACKED_MODELS = ["LinearRegression", "LightGBM", "Baseline (mean)"]
ANALYSIS_MODELS = ["LinearRegression", "LightGBM"]
# Maps model label → training function for ablation re-training.
TRAIN_FUNCS = {
    "LinearRegression": train_linear_regression,
    "LightGBM": train_lightgbm,
}


def _evaluate(model, X_test, Y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(Y_test, preds)),
        "r2": float(r2_score(Y_test, preds)),
    }


def _aggregate(folds: list[dict]) -> list[dict]:
    """Mean ± stddev of MAE and R² per model across folds, plus worst-fold MAE.
    Surfacing min/max distinguishes "reliably mediocre" from "great most folds,
    bad on one"."""
    out = []
    for label in TRACKED_MODELS:
        maes = [m["mae"] for f in folds for m in f["models"] if m["label"] == label]
        r2s = [m["r2"] for f in folds for m in f["models"] if m["label"] == label]
        out.append(
            {
                "label": label,
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes, ddof=1)),
                "mae_min": float(np.min(maes)),
                "mae_max": float(np.max(maes)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s, ddof=1)),
            }
        )
    return out


def _aggregate_importance(per_fold: dict[str, list[pd.Series]]) -> list[dict]:
    """Mean importance across folds, per model."""
    out = []
    for label, importances in per_fold.items():
        mean = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        out.append(
            {
                "label": label,
                "importance": [{"feature": f, "value": float(v)} for f, v in mean.items()],
            }
        )
    return out


def _aggregate_ablation(per_fold: dict[str, list[pd.Series]]) -> list[dict]:
    """Mean and stddev of ablation Δ per feature across folds. Stddev gives
    a sense of how stable the contribution is across folds."""
    out = []
    for label, deltas_per_fold in per_fold.items():
        stacked = pd.concat(deltas_per_fold, axis=1)
        mean = stacked.mean(axis=1).sort_values(ascending=False)
        std = stacked.std(axis=1, ddof=1)
        out.append(
            {
                "label": label,
                "ablation": [
                    {"feature": f, "delta_mean": float(mean[f]), "delta_std": float(std[f])}
                    for f in mean.index
                ],
            }
        )
    return out


def main():
    df = load_player_data()

    fold_results: list[dict] = []
    per_fold_importance: dict[str, list[pd.Series]] = {label: [] for label in ANALYSIS_MODELS}
    per_fold_ablation: dict[str, list[pd.Series]] = {label: [] for label in ANALYSIS_MODELS}

    for fold in WALK_FORWARD_FOLDS:
        X_train, Y_train, X_test, Y_test = build_training_set(
            df, train_seasons=fold["train"], test_seasons=fold["test"]
        )

        lr = train_linear_regression(X_train, Y_train)
        lgbm = train_lightgbm(X_train, Y_train)
        baseline = DummyRegressor(strategy="mean").fit(X_train, Y_train)

        fold_models = [
            {"label": "LinearRegression", **_evaluate(lr, X_test, Y_test)},
            {"label": "LightGBM", **_evaluate(lgbm, X_test, Y_test)},
            {"label": "Baseline (mean)", **_evaluate(baseline, X_test, Y_test)},
        ]
        fold_results.append(
            {
                "test_season": fold["test"][0],
                "train_seasons": fold["train"],
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "models": fold_models,
            }
        )

        for label, trained_model in [("LinearRegression", lr), ("LightGBM", lgbm)]:
            per_fold_importance[label].append(feature_importance(trained_model, X_test, Y_test))
            per_fold_ablation[label].append(
                feature_ablation(TRAIN_FUNCS[label], X_train, Y_train, X_test, Y_test)
            )

    aggregate = _aggregate(fold_results)
    importance_results = _aggregate_importance(per_fold_importance)
    ablation_results = _aggregate_ablation(per_fold_ablation)

    Path("metrics.json").write_text(
        json.dumps({"folds": fold_results, "aggregate": aggregate}, indent=2)
    )
    Path("importance.json").write_text(json.dumps(importance_results, indent=2))
    Path("ablation.json").write_text(json.dumps(ablation_results, indent=2))

    # Summary
    print("Walk-forward CV (sliding window, 3 train seasons per fold)")
    print("=" * 60)
    for fold in fold_results:
        print(
            f"\n[Test {fold['test_season']}  train_rows={fold['train_rows']}  "
            f"test_rows={fold['test_rows']}]"
        )
        for m in fold["models"]:
            print(f"  {m['label']:18} MAE={m['mae']:6.2f}  R²={m['r2']:6.2f}")

    print("\n" + "=" * 60)
    print("Aggregate (mean ± stddev across folds)")
    print("=" * 60)
    for m in aggregate:
        print(
            f"  {m['label']:18} MAE={m['mae_mean']:6.2f} ± {m['mae_std']:.2f}   "
            f"R²={m['r2_mean']:5.2f} ± {m['r2_std']:.2f}"
        )

    # Combined permutation + ablation per model, ordered by ablation Δ.
    for label in ANALYSIS_MODELS:
        imp = next(e for e in importance_results if e["label"] == label)
        abl = next(e for e in ablation_results if e["label"] == label)
        imp_lookup = {i["feature"]: i["value"] for i in imp["importance"]}

        print(f"\n{label} feature analysis (mean across folds):")
        print(f"  {'feature':30} {'permutation':>12} {'ablation Δ':>13}")
        for item in abl["ablation"]:
            f = item["feature"]
            print(f"  {f:30} {imp_lookup.get(f, 0.0):12.2f} {item['delta_mean']:+13.2f}")


if __name__ == "__main__":
    main()

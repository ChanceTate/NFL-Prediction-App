# NFL Prediction App

Full-stack machine learning project for predicting NFL quarterback passing yards from historical player data. The app builds a quarterback-only training set, engineers matchup and recent-form features, compares multiple regression models, and reports walk-forward validation metrics.

## Overview

Quarterback passing performance changes week to week, so the model uses prior-game context instead of current-game results. The pipeline:

- downloads and caches NFL player statistics with `nflreadpy`
- filters the dataset to quarterbacks
- creates rolling and matchup features from historical games
- trains Linear Regression, LightGBM, and a mean baseline model
- evaluates each model with sliding-window walk-forward validation
- writes model metrics and feature analysis artifacts

## Tech Stack

- Python 3.12+
- pandas
- scikit-learn
- LightGBM
- nflreadpy
- pytest and ruff
- GitHub Actions for linting, tests, and model metrics

## Data

Player stats are loaded from `nflreadpy` and cached under `data/` so repeat runs do not have to download the same seasons again. Each row represents one player game and includes passing, rushing, receiving, team, opponent, season, and week fields. The training pipeline filters this data to QB rows after using the full dataset for team, defense, and receiver context.

## Feature engineering

The current feature set includes:

- rolling 3-game QB passing yards
- rolling 3-game QB pass attempts
- opponent rolling passing yards allowed
- rolling EPA per pass attempt
- rolling team offensive plays
- top receiver rolling receiving yards
- QB historical average against the same defense
- rolling passing-yards trend slope

These features are generated without using the current game's target value, then checked for unexpected missing values before training.

## Model evaluation

The project evaluates models with sliding-window walk-forward validation. Each fold trains on three consecutive seasons and tests on the following season, from 2019 through 2025 test seasons. Results are reported with MAE and R² for:

- Linear Regression
- LightGBM
- Baseline mean regressor

Running `main.py` also produces:

- `metrics.json` with per-fold and aggregate metrics
- `importance.json` with permutation feature importance
- `ablation.json` with leave-one-feature-out ablation results

## Model metrics in pull requests

The CI workflow runs the model on pull requests, uploads the metrics artifacts, and compares PR results against the latest successful `main` run when available. This gives reviewers an automatically updated view of whether a change improves or regresses model performance.

## Getting started

Install dependencies with uv:

```bash
uv sync --all-groups
```

Run linting and tests:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Run the model pipeline:

```bash
uv run main.py
```

## Docker

The included Dockerfile can be used for reproducible local runs or deployment environments that should not depend on a developer's Python installation.

# Feature Experiments Log

Every feature tested with ablation: what it was, how it scored, and the verdict.

**Decision rule (two paths):** keep a feature if, in either model, it passes **either**:
- *Magnitude path:* mean ablation Δ ≥ +0.2 with 5+/7 folds positive
- *Consistency path:* mean ablation Δ ≥ +0.1 with 6+/7 folds positive

Sign consistency (folds+) is the cleanest evidence the effect is real;
magnitude is the "is it big enough to care about" check. The two paths separate
those questions so reliable-but-marginal features can earn their keep without
having to clear the harder magnitude bar alone.

## Rejected

| Feature | LR Δ (folds+) | LGBM Δ (folds+) | Why |
|---|---|---|---|
| rolling QB rushing attempts | -0.05 (3/7) | -0.08 (3/7) | redundant with `rolling_pass_atts_3` (mobile QBs throw less, already captured) |
| rolling variance/std of passing yards | +0.02 (3/7) | -0.07 (3/7) | second-order noise |
| rolling fantasy points | +0.06 (4/7) | +0.07 (4/7) | redundant with `rolling_yds_3` (dominated by passing yards for QBs) |
| rolling air yards per attempt | +0.00 (4/7) | +0.05 (3/7) | throw-depth absorbed by other features |
| rolling air yards (volume) | -0.03 (2/7) | +0.12 (4/7) | same signal as the rate version, also rejected |
| defense YPA allowed | +0.04 (5/7) | +0.04 (5/7) | refines an already-weak defense signal |
| acceleration (slope of slope) | -0.04 (1/7) | +0.04 (4/7) | hurts LR; cannibalizes `rolling_yds_slope_3` in LGBM |
| rolling sacks suffered | -0.05 (3/7) | +0.07 (5/7) | doesn't clear +0.2 in either model |
| rolling completion percentage | +0.01 (4/7) | +0.06 (4/7) | redundant with EPA per attempt |
| rolling CPOE (volume-weighted) | -0.04 (2/7) | +0.09 (4/7) | efficiency rate, collides with `rolling_epa_per_att_3` |
| rolling home/away share | -0.05 (3/7) | +0.01 (3/7) | weak signal even after fixing the game_id join bug |

## Kept

| Feature | What it captures |
|---|---|
| `rolling_yds_3` | QB recent passing yards |
| `rolling_pass_atts_3` | QB recent pass volume |
| `opp_pass_yds_allowed_3` | defense recent yards allowed |
| `rolling_epa_per_att_3` | QB recent passing efficiency |
| `rolling_team_plays_3` | team pace |
| `top_receiver_rolling_yds_3` | best receiver's recent yards |
| `qb_vs_def_avg_yds` | this QB vs this defense matchup history |
| `rolling_yds_slope_3` | direction of recent passing yards trend |
| `last_game_vs_season_avg` | gap from QB's season-to-date avg |
| `rolling_pass_fd_per_att_3` | drive sustainability. first downs earned through the air, per attempt |

## Current state

- LR: 68.3 MAE ± 2.2
- LGBM: 67.3 MAE ± 2.2
- Baseline (mean): 81.0 MAE ± 3.5
- Noise floor estimate without Vegas: ~50-55 MAE

KBO PLAYER NEXT-SEASON PERFORMANCE: EXPERIMENT REPORT
=====================================================
Scope: qualified players only (hitters >=100 PA; pitchers >=30 IP).
IMPORTANT: qualification includes TARGET season; results are conditional on future playing time.
Rows represent player-season observations, not necessarily distinct players.
2024 validation: tuning/selection; 2025 test: held-out evaluation only.
Do not select a new model by looking for the lowest 2025 test error.
Batter wRC+ and pitcher FIP MAEs have DIFFERENT units; do not compare directly.
v3 native lag sizes differ; only v3 common cohort isolates history-length effects.
Small common-cohort test sizes mean apparent improvements may be noisy.
Ablation reflects predictive contribution, not causal effects of pitcher role.

V2 MODEL COMPARISON
-------------------
type=batter | model=ridge | validation MAE=15.9016
type=pitcher | model=ridge | validation MAE=0.7766

V3 COMMON HISTORY COMPARISON
----------------------------
type=batter | cohort=common | model=ridge | validation MAE=14.1664 | lag=1
type=pitcher | cohort=common | model=random_forest | validation MAE=0.9423 | lag=1

V4 ROLE ABLATION
----------------
type=pitcher | cohort=native | lag=1 | model=ridge | validation MAE=0.7765 | variant=no_gs_rate


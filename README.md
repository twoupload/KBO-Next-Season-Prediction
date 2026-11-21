
# KBO Next-Season Performance Prediction

## 1. Project Overview

본 프로젝트는 KBO 선수의 과거 시즌 기록을 이용하여
차기 시즌 성과를 예측하는 컴퓨터공학 졸업논문 연구입니다.

- 타자 예측 목표: wRC+
- 투수 예측 목표: FIP
- 데이터 출처: FanGraphs KBO
- 분석 기간: 2015–2025
- 추가 평가: 2026년 정규시즌

## 2. Prediction Methods

### Machine Learning Models
- Ridge Regression
- Random Forest
- XGBoost

### Baselines
- Previous-season persistence
- League-average baseline
- Marcel-inspired baseline

Marcel-inspired baseline은 원본 MLB Marcel 시스템을
그대로 재현한 것이 아니라 KBO 데이터와 예측 목표에
맞게 변형한 방법입니다.

## 3. Experimental Design

- Training: Target seasons through 2023
- Validation: 2024
- Independent test: 2025
- Additional evaluation: 2026

Evaluation criteria:
- Hitters: At least 100 PA
- Pitchers: At least 30 IP

Evaluation metrics:
- MAE
- RMSE
- R²

Additional analyses:
- Lag1–Lag3 feature history comparison
- Pitcher GS / GS_rate ablation
- Paired bootstrap (10,000 resamples)
- Marcel parameter sensitivity analysis

## 4. Main Results

### 2025 Independent Test

| Target | Best Method | MAE |
|---|---|---|
| Hitter wRC+ | XGBoost | 16.915 |
| Pitcher FIP | Marcel-inspired | 0.684 |

The performance difference between XGBoost and
Ridge for hitters was inconclusive under paired
bootstrap analysis.

For pitchers, the Marcel-inspired baseline
outperformed Ridge in the 2025 test sample.

## 5. Repository Structure

- build_kbo_season_links.py: Data preprocessing
- train_model_v2.py: Main model comparison
- train_model_v3.py: History-length experiments
- train_model_v4.py: Pitcher role-feature ablation
- predict_2026.py: 2026 predictions
- evaluate_2026.py: 2026 evaluation
- compare_baselines.py: Statistical baselines
- analyze_significance.py: Bootstrap analysis
- analyze_marcel_sensitivity.py: Sensitivity analysis
- analyze_model_results.py: Result analysis
- make_thesis_figures.py: Figure generation

## 6. Installation

Python 3.x is required.

```bash
python -m pip install -r requirements.txt
```

## 7. Data Availability

The original KBO statistics were obtained from
FanGraphs.

https://www.fangraphs.com/

Raw data redistribution is subject to the
original provider's terms.

Required input files and preparation instructions
will be documented after verifying the full
execution workflow.

## 8. Reproducibility

Model hyperparameters were selected using
2024 validation MAE.

The 2025 independent test was not used for
hyperparameter selection.

The full execution order and environment versions
will be finalized after verification.

## 9. Research Limitations

- Predictions are evaluated on qualified players.
- Extreme performance changes are difficult to predict.
- Statistical baselines can outperform machine learning.
- Performance may vary across evaluation seasons.



# Experimental Results

이 폴더는 KBO 선수 차기 시즌 성과 예측 연구의
주요 실험 결과를 보관합니다.

## Main Experiments

1. 2025 Independent Test
   - Ridge, Random Forest, XGBoost
   - Persistence, League Mean, Marcel-inspired

2. Feature History Comparison
   - Lag1, Lag2, Lag3

3. Pitcher Feature Ablation
   - GS and GS_rate

4. 2026 Additional Evaluation
   - Hitters: 109 players
   - Pitchers: 83 players

5. Statistical Validation
   - Paired bootstrap
   - 10,000 resamples
   - 95% confidence intervals

6. Marcel Sensitivity Analysis
   - Regression strength
   - Age adjustment

## Main Results

2025 independent test:

- Hitter wRC+: XGBoost MAE = 16.915
- Pitcher FIP: Marcel-inspired MAE = 0.684

2026 additional evaluation:

- Hitter Ridge MAE = 19.206
- Pitcher Ridge MAE = 0.663

## Reproducibility

각 CSV는 논문에 제시된 결과를 확인하기 위한
자료입니다.

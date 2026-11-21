# KBO Next-Season Performance Prediction

머신러닝을 활용한 KBO 선수 차기 시즌 성과 예측 연구(컴퓨터공학 졸업논문)입니다. 과거 시즌의 기록을 사용해 **타자 wRC+**와 **투수 FIP**를 예측하고, 단순 기준선 및 통계적 기준선과 비교합니다.

## 1. 연구 개요

- **데이터 출처:** [FanGraphs](https://www.fangraphs.com/) KBO 선수별 시즌 기록
- **과거 기록:** 2015–2025년
- **학습·검증·테스트:** 2023년까지 학습, 2024년 검증, 2025년 독립 테스트
- **추가 시즌 평가:** 2026년 정규시즌 결과
- **평가 지표:** MAE, RMSE, R² (MAE와 RMSE는 낮을수록 좋음)
- **평가 표본:** 해당 평가 시즌에 타자 100 PA 이상, 투수 30 IP 이상인 선수

**표본 해석 주의:** 학습 및 평가용 `qualified` 자료는 **예측 대상 시즌의 실제 출전량**으로 표본을 제한합니다. 따라서 결과는 미래 출전량 기준을 충족한 선수에 대한 *조건부 예측 성능*이며, 시즌 시작 전에 모든 선수를 대상으로 예측한 성능과 동일하지 않습니다. 선수-시즌 관측치 수와 고유 선수 수 역시 구분해야 합니다.

## 2. 예측 방법 및 실험

**머신러닝 모델:** Ridge Regression, Random Forest, XGBoost

**비교 기준선:** 직전 시즌 성적 유지(Persistence), 리그 평균(League mean), Marcel-inspired

Marcel-inspired 기준선은 원본 MLB Marcel 시스템의 완전한 구현이 아니라 KBO 데이터와 wRC+/FIP 예측에 맞춘 변형 방식입니다.

- **V2:** 기본 모델 및 하이퍼파라미터 비교
- **V3:** 과거 1·2·3시즌 입력(Lag1–Lag3) 비교. 표본 크기가 달라지는 `native`와 동일 선수-시즌을 비교하는 `common`을 구분
- **V4:** 투수의 `GS_rate`, `GS` 입력 변수 제거 실험
- **통계 분석:** 선수별 쌍대 부트스트랩(10,000회), Marcel 기준선의 회귀 강도·연령 보정 민감도 분석

하이퍼파라미터는 **2024년 검증 MAE**를 기준으로 선택하며, 2025년 테스트 및 2026년 추가 평가 결과로 재선택하지 않습니다. 구체적인 탐색 범위와 선택 결과는 `train_model_v2.py`, `train_model_v3.py`, `train_model_v4.py` 및 `results/ml_results_v2/` 등의 실험 기록에서 확인할 수 있습니다.

## 3. 주요 결과

### 2025년 독립 테스트

| 예측 대상 | 해당 대상의 최저 MAE 방법 | MAE | 평가 관측치 |
|---|---|---:|---:|
| 타자 wRC+ | XGBoost | 16.9152 | 106 |
| 투수 FIP | Marcel-inspired | 0.6839 | 86 |

2025년 타자 XGBoost와 Ridge의 MAE 차이는 작습니다(Ridge: 16.9253). 이 결과만으로 XGBoost의 우월성을 단정하지 않습니다. 투수에서는 Marcel-inspired 기준선이 Ridge(MAE 0.7556)보다 낮은 MAE를 기록했습니다. 위 표는 **2025년 결과의 최저 MAE를 설명**하는 것이며, 2025년 결과로 모델을 새로 선택했다는 뜻은 아닙니다.

### 2026년 추가 평가 (Ridge)

| 예측 대상 | 예측 인원 | 평가 인원 | 제외 인원 | Ridge MAE | 직전 시즌 유지 MAE | MAE 감소율 |
|---|---:|---:|---:|---:|---:|---:|
| 타자 wRC+ | 155 | 109 | 46 | 19.2057 | 23.9144 | 19.69% |
| 투수 FIP | 138 | 83 | 55 | 0.6625 | 0.8348 | 20.64% |

2026년 평가 대상은 총 **192명**(타자 109명, 투수 83명)입니다. 제외 사유에는 출전량 미달 또는 평가 자료에서의 기록 부재 등이 포함됩니다. 2026년 결과는 과거 자료만 사용해 생성한 예측의 **추가 시즌 검증**이며, 사전에 등록된 실시간 예측 실험이라고 주장하지 않습니다. 타자 wRC+와 투수 FIP의 절대 MAE는 단위가 다르므로 직접 비교할 수 없습니다.

## 4. 저장소 구성

```text
.
├── data/
│   ├── kbo_batter_2015_2025_active.csv
│   ├── kbo_pitcher_2015_2025_active.csv
│   ├── 2026_total_batter.csv
│   ├── 2026_total_pitcher.csv
│   └── kbo_linked_2015_2025/
├── results/
│   ├── ml_results_v2/
│   ├── ml_results_v3/
│   ├── ml_results_v4/
│   ├── predictions_2026/
│   ├── evaluation_2026/
│   ├── baseline_comparison/
│   ├── significance_results/
│   ├── marcel_sensitivity_results/
│   ├── analysis_results/
│   └── thesis_figures/
├── build_kbo_season_links.py
├── train_model_v2.py
├── train_model_v3.py
├── train_model_v4.py
├── predict_2026.py
├── evaluate_2026.py
├── compare_baselines.py
├── analyze_significance.py
├── analyze_marcel_sensitivity.py
├── analyze_model_results.py
├── make_thesis_figures.py
├── requirements.txt
└── README.md
```

`data/`에는 원본·전처리 데이터가, `results/`에는 모델별 예측값·평가표·분석 결과·그림이 저장됩니다. 자세한 설명은 각 폴더의 README를 참고하세요.

## 5. 설치 및 실행 순서

프로젝트 최상위 폴더에서 실행합니다. Python 3 환경이 필요합니다.

```powershell
python -m pip install -r requirements.txt
```

처음부터 전체 실험을 재생성하려면 다음 순서대로 실행합니다.

```powershell
# 1. 선수-시즌 연결 및 입력 데이터 생성
python build_kbo_season_links.py

# 2. 기본 모델 및 추가 실험
python train_model_v2.py
python train_model_v3.py
python train_model_v4.py

# 3. 2026년 예측과 평가
python predict_2026.py
python evaluate_2026.py

# 4. 기준선 및 통계 분석
python compare_baselines.py
python analyze_significance.py
python analyze_marcel_sensitivity.py

# 5. 분석표 및 논문용 그림 생성
python analyze_model_results.py
python make_thesis_figures.py
```

각 단계는 `data/` 또는 이전 단계에서 생성된 `results/`의 파일을 사용합니다. **2026년 실제 기록 파일은 예측 생성(`predict_2026.py`)에는 사용되지 않고, 이후 평가(`evaluate_2026.py`)에서 사용됩니다.**

이미 결과 파일이 포함된 저장소에서는 필요한 단계만 재실행할 수 있습니다. `requirements.txt`에는 패키지 이름이 기재되어 있으며 버전이 고정되어 있지 않으므로, 실행 환경에 따라 미세한 수치 차이가 발생할 수 있습니다. Windows에서 한글 그래프 출력에는 `Malgun Gothic` 글꼴을 사용합니다.

## 6. 데이터 출처와 재현성

통계 자료는 FanGraphs에서 수집했습니다. 원본 자료의 이용 및 재배포는 제공자의 약관을 확인해야 합니다. 실험 재현에 필요한 입력 파일은 `data/`에서 확인할 수 있으며, 각 실험의 출력 CSV와 그림은 `results/`에 보관합니다.

- 2024년 검증: 하이퍼파라미터 탐색 및 선택
- 2025년 독립 테스트: 선택된 설정의 성능 평가
- 2026년 추가 평가: 2015–2025년 과거 자료 기반 예측과 2026년 실제 성적 비교
- 부트스트랩: 선수별 대응 관계를 유지한 MAE 차이 분석

## 7. 해석상의 한계

1. **미래 출전량에 따른 표본 선택:** `qualified` 평가 결과는 미래 시즌의 출전량 기준을 충족한 선수에 한정됩니다.
2. **극단값 예측의 어려움:** 성적이 급격히 개선·악화되는 선수를 정확히 예측하기 어렵고, 예측값이 평균 부근으로 수축할 수 있습니다.
3. **기준선의 경쟁력:** 일부 시즌·대상에서는 Marcel-inspired와 같은 통계적 기준선이 머신러닝보다 낮은 오차를 기록합니다.
4. **표본과 시즌의 제약:** 연도별 선수 구성과 표본 크기가 다르며, V3의 과거 시즌 수 비교는 `common` 표본을 우선 해석해야 합니다.
5. **인과 해석 불가:** V4 변수 제거 결과는 예측 성능의 변화이지 투수 보직의 인과효과를 뜻하지 않습니다.

---

본 저장소는 KBO 선수 차기 시즌 성과 예측에 관한 졸업논문의 코드, 데이터 처리 절차 및 실험 결과를 정리한 연구 자료입니다.

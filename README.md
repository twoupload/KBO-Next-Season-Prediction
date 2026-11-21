
# KBO Next-Season Performance Prediction

머신러닝을 활용한 KBO 선수 차기 시즌 성과 예측 연구(컴퓨터공학 졸업논문)입니다.

과거 시즌의 기록을 사용해 **타자 wRC+**와 **투수 FIP**를 예측하고, 머신러닝 모델과 통계적 기준선의 성능을 비교합니다.

## 1. 연구 개요

- **데이터 출처:** [FanGraphs](https://www.fangraphs.com/) KBO 선수별 시즌 기록
- **과거 기록:** 2015~2025년
- **학습 기간:** 2023년까지
- **검증 기간:** 2024년
- **독립 테스트:** 2025년
- **추가 시즌 평가:** 2026년 정규시즌
- **평가 지표:** MAE, RMSE, R²
- **평가 기준:** 타자 100 PA 이상, 투수 30 IP 이상

### 평가 표본에 관한 주의사항

학습 및 평가에 사용하는 `qualified` 데이터는 예측 대상 시즌의 실제 출전량을 기준으로 표본을 제한합니다.

따라서 본 연구의 평가 결과는 **미래 시즌의 출전량 기준을 충족한 선수에 대한 조건부 예측 성능**입니다.

시즌 시작 시점에 모든 선수를 대상으로 수행한 예측의 성능과 동일하게 해석할 수 없습니다.

또한 학습 데이터의 선수-시즌 관측치 수와 고유 선수 수는 구분해야 합니다.

## 2. 예측 모델 및 실험 구성

### 2.1 머신러닝 모델

다음 세 가지 회귀 모델을 사용합니다.

- Ridge Regression
- Random Forest
- XGBoost

### 2.2 비교 기준선

머신러닝 모델의 성능을 평가하기 위해 다음 기준선을 사용합니다.

- **Persistence:** 직전 시즌 성적을 다음 시즌 예측값으로 사용
- **League Mean:** 과거 리그 평균에 기반한 예측
- **Marcel-inspired:** 과거 성적의 가중평균, 리그 평균 회귀, 연령 보정을 적용한 예측

Marcel-inspired 기준선은 원본 MLB Marcel 시스템을 완전히 재현한 것이 아니라, KBO 데이터와 wRC+/FIP 예측에 맞게 변형한 방식입니다.

### 2.3 추가 실험

| 실험 | 목적 |
|---|---|
| V2 | 기본 머신러닝 모델 및 하이퍼파라미터 비교 |
| V3 | 과거 1·2·3시즌 입력에 따른 예측 성능 비교 |
| V4 | 투수 보직 관련 변수 제거에 따른 성능 변화 분석 |
| 통계 분석 | 쌍대 부트스트랩을 통한 MAE 차이 검토 |
| 민감도 분석 | Marcel-inspired 기준선의 설정 변화 분석 |

V3에서는 과거 시즌 수에 따라 표본 크기가 달라지는 `native`와 동일한 선수-시즌을 비교하는 `common`을 구분합니다.

V4에서는 투수의 `GS_rate` 및 `GS` 변수를 제거하여 예측 성능 변화를 관찰합니다.

모델의 하이퍼파라미터는 **2024년 검증 MAE**를 기준으로 선택합니다.

2025년 테스트 및 2026년 추가 평가 결과는 하이퍼파라미터 재선택에 사용하지 않습니다.

## 3. 주요 실험 결과

### 3.1 2025년 독립 테스트

| 예측 대상 | 최저 MAE 방법 | MAE | 평가 관측치 |
|---|---|---:|---:|
| 타자 wRC+ | XGBoost | 16.9152 | 106 |
| 투수 FIP | Marcel-inspired | 0.6839 | 86 |

2025년 타자 예측에서는 XGBoost가 가장 낮은 MAE를 기록했으며, Ridge의 MAE는 16.9253으로 두 모델의 차이가 작았습니다.

투수 예측에서는 Marcel-inspired 기준선이 Ridge(MAE 0.7556)보다 낮은 MAE를 기록했습니다.

위 결과는 2025년 테스트에서 관측된 성능을 설명한 것이며, 해당 결과를 사용하여 모델을 새로 선택했다는 뜻은 아닙니다.

### 3.2 2026년 추가 평가

2015~2025년의 과거 기록으로 생성한 Ridge 예측값을 2026년 정규시즌 실제 성적과 비교했습니다.

| 구분 | 타자 wRC+ | 투수 FIP |
|---|---:|---:|
| 전체 예측 대상 | 155명 | 138명 |
| 최종 평가 대상 | 109명 | 83명 |
| 평가 제외 | 46명 | 55명 |
| Ridge MAE | 19.2057 | 0.6625 |
| Persistence MAE | 23.9144 | 0.8348 |
| MAE 감소율 | 19.69% | 20.64% |

전체 예측 대상은 293명이며, 이 중 192명이 최종 평가 기준을 충족했습니다.

평가 제외 사유에는 출전량 미달과 평가 자료에서의 기록 부재 등이 포함됩니다.

Ridge 모델은 Persistence 기준선과 비교했을 때 타자에서 약 19.69%, 투수에서 약 20.64% 낮은 MAE를 기록했습니다.

단, 타자 wRC+와 투수 FIP는 서로 다른 지표이므로 MAE의 절댓값을 직접 비교할 수 없습니다.

또한 2026년 결과는 추가 시즌 평가이며, 사전에 등록된 실시간 예측 실험이라고 주장하지 않습니다.

## 4. 프로젝트 폴더 구조

```text
KBO-Next-Season-Prediction/
│
├── data/
│   ├── kbo_batter_2015_2025_active.csv
│   ├── kbo_pitcher_2015_2025_active.csv
│   ├── 2026_total_batter.csv
│   ├── 2026_total_pitcher.csv
│   └── kbo_linked_2015_2025/
│
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
│
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

`data/`에는 원본 및 전처리 데이터가 저장되며, `results/`에는 모델 학습 결과, 예측값, 평가표, 통계 분석 결과와 논문용 그림이 저장됩니다.

## 5. 설치 및 실행 방법

### 5.1 실행 환경

Python 3 환경에서 실행합니다.

필요한 라이브러리는 다음과 같습니다.

- NumPy
- pandas
- scikit-learn
- XGBoost
- Matplotlib

프로젝트 최상위 폴더에서 다음 명령어를 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

### 5.2 데이터 전처리

선수별 시즌 기록을 연결하여 모델 학습에 사용할 데이터를 생성합니다.

```powershell
python build_kbo_season_links.py
```

전처리 결과는 `data/kbo_linked_2015_2025/`에 저장됩니다.

### 5.3 머신러닝 모델 학습

기본 모델 및 추가 실험을 실행합니다.

```powershell
python train_model_v2.py
python train_model_v3.py
python train_model_v4.py
```

각 실험 결과는 다음 경로에 저장됩니다.

- `results/ml_results_v2/`
- `results/ml_results_v3/`
- `results/ml_results_v4/`

### 5.4 2026년 성적 예측

2015~2025년 데이터를 사용하여 2026년 선수별 성적을 예측합니다.

```powershell
python predict_2026.py
```

예측 결과는 `results/predictions_2026/`에 저장됩니다.

이 단계에서는 2026년 실제 성적을 모델 입력에 사용하지 않습니다.

### 5.5 2026년 실제 성적 평가

생성된 예측값과 2026년 정규시즌 실제 기록을 비교합니다.

```powershell
python evaluate_2026.py
```

평가 결과는 `results/evaluation_2026/`에 저장됩니다.

### 5.6 기준선 및 통계 분석

다음 명령어를 순서대로 실행합니다.

```powershell
python compare_baselines.py
python analyze_significance.py
python analyze_marcel_sensitivity.py
```

각 결과는 다음 경로에 저장됩니다.

- `results/baseline_comparison/`
- `results/significance_results/`
- `results/marcel_sensitivity_results/`

쌍대 부트스트랩 분석은 선수별 대응 관계를 유지한 상태에서 10,000회 재표본추출을 수행합니다.

### 5.7 종합 분석 및 논문용 그림 생성

```powershell
python analyze_model_results.py
python make_thesis_figures.py
```

생성된 결과는 다음 경로에서 확인할 수 있습니다.

- `results/analysis_results/`
- `results/thesis_figures/`

Windows 환경에서 한글 그래프 출력에는 `Malgun Gothic` 글꼴을 사용합니다.

## 6. 실험 재현성

본 프로젝트에서는 시간 순서에 따른 데이터 분할을 적용합니다.

| 구분 | 사용 기간 | 목적 |
|---|---|---|
| 학습 | 2023년까지 | 모델 학습 |
| 검증 | 2024년 | 하이퍼파라미터 탐색 및 선택 |
| 독립 테스트 | 2025년 | 선택된 모델의 성능 평가 |
| 추가 평가 | 2026년 | 과거 기록 기반 예측의 추가 검증 |

입력 변수는 과거 시즌 기록으로 구성하며, 예측 대상 시즌의 실제 성적을 입력 변수로 사용하지 않습니다.

각 모델의 하이퍼파라미터 탐색 범위와 선택 결과는 학습 코드 및 `results/`의 실험 기록에서 확인할 수 있습니다.

`requirements.txt`에는 패키지 이름이 기재되어 있으나 정확한 버전이 고정되어 있지 않습니다. 따라서 실행 환경에 따라 일부 수치 차이가 발생할 수 있습니다.

## 7. 연구의 한계

### 7.1 미래 출전량에 따른 표본 선택

`qualified` 데이터는 예측 대상 시즌의 실제 출전량을 기준으로 평가 표본을 제한합니다.

따라서 전체 선수 집단에 대한 무조건적인 예측 성능으로 일반화할 수 없습니다.

### 7.2 극단적인 성적 변화 예측의 어려움

예측값이 평균 부근으로 수축하는 경향이 나타날 수 있으며, 전년도 대비 성적이 급격하게 상승하거나 하락하는 선수의 예측 오차가 커질 수 있습니다.

### 7.3 통계적 기준선의 경쟁력

일부 시즌에서는 Marcel-inspired와 같은 통계적 기준선이 머신러닝 모델보다 낮은 오차를 기록했습니다.

따라서 복잡한 머신러닝 모델이 항상 더 높은 예측 성능을 보장하는 것은 아닙니다.

### 7.4 표본 크기 및 시즌별 차이

선수 구성과 표본 크기는 시즌에 따라 달라집니다.

특히 과거 시즌 수를 비교하는 V3 실험에서는 동일한 선수-시즌으로 구성한 `common` 표본의 결과를 우선적으로 해석해야 합니다.

### 7.5 변수 제거 실험의 해석

V4 실험은 투수 보직 관련 변수가 예측 성능에 미치는 영향을 관찰하기 위한 것입니다.

변수 제거에 따른 성능 차이는 인과관계를 입증하는 결과가 아닙니다.

## 8. 데이터 출처

- [FanGraphs](https://www.fangraphs.com/) — KBO 선수별 시즌 기록

원본 데이터의 이용 및 재배포는 제공자의 이용 약관을 확인해야 합니다.

---

본 저장소는 머신러닝을 활용한 KBO 선수 차기 시즌 성과 예측 졸업논문의 코드, 데이터 처리 과정, 실험 결과 및 재현 절차를 정리한 연구 자료입니다.

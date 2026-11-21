KBO 기준선 추가 실험 (2026-11-17)

실행: python compare_baselines.py
필요 패키지: pandas, numpy, scikit-learn
실행 위치: 기존 kbo_work 프로젝트 폴더

데이터: data/kbo_{batter,pitcher}_2015_2025_active.csv
비교 대상: ml_results_v2/*_{validation,test}_predictions.csv 및 evaluation_2026/*_2026_qualified_comparison.csv

방식:
- 리그 평균: 예측 시점 이전 최대 3개 시즌의 PA/IP 가중 지표 평균을 5/4/3으로 결합
- Marcel 변형: 선수별 이전 3개 시즌의 5/4/3 및 PA/IP 가중 성적과 위 리그 평균을 결합
- 평균 회귀 강도: 타자 1200 PA, 투수 1080 outs (=360 IP). 투수의 값은 원래 Marcel의 공식 설정이 아닌 연구상 고정된 적응값
- 연령 보정: 예측 연령 29세 이하 0.006, 초과 0.003의 구간 기울기를 사용해 리그 평균으로부터의 편차에 곱함. 원본 Marcel의 지표별 연령 보정과 다름.
- 타자 wRC+ 및 투수 FIP에 적용한 'Marcel-inspired adaptation'이며 원본 Marcel 재현으로 기술하면 안 됨.
- 하이퍼파라미터는 2025/2026 결과를 보고 조정하지 않았음.
- 2015년 이전 기록은 없어 2016/2017년 학습용 투영 시 최대 1/2시즌만 사용 가능. 이번 평가 대상 2024/2025/2026에는 3시즌 범위 사용 가능.
- 리그 평균은 제공된 FanGraphs 통합 CSV에서 계산한 출전량 가중 평균이며 공식 KBO 리그 평균과 정확히 일치한다고 보장할 수 없음.
- 기존 2026년 예측 선수 전체가 아닌, 기존 2026년 평가 자격을 충족한 동일 선수에 대한 사후 성능 비교임.
- Marcel 변형 방식의 투수 평균 회귀 강도 및 연령 보정은 추가적인 사전 정의와 정당화가 필요한 연구 설계 선택. 최종 논문에는 공식, 선택 이유, 한계를 명시할 것.

출처: Tom Tango 원문 https://www.tangotiger.net/archives/stud0214.shtml ; https://www.tangotiger.net/archives/stud0346.shtml

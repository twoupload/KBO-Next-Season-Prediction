
# Dataset Information

## Data Source

본 연구는 FanGraphs에서 제공하는 KBO 선수별
시즌 통계 자료를 활용합니다.

Source: https://www.fangraphs.com/

## Data Period

- Historical records: 2015–2025
- Additional evaluation: 2026

## Prediction Targets

- Batters: wRC+
- Pitchers: FIP

## Evaluation Criteria

- Batters: 100 PA or more
- Pitchers: 30 IP or more

## Data Preparation

1. FanGraphs에서 KBO 선수별 시즌 기록을 수집합니다.
2. 타자와 투수 기록을 구분합니다.
3. PlayerId와 Season을 기준으로 기록을 연결합니다.
4. 직전 1~3시즌의 입력 특징을 생성합니다.
5. 평가 기준에 따라 선수 표본을 구성합니다.

"""Marcel-inspired baseline sensitivity analysis for KBO.

Place this script next to batter_2024_comparison.csv, ...,
pitcher_2026_comparison.csv in baseline_comparison/.
Run: python analyze_marcel_sensitivity.py
Dependencies: pandas, numpy

This script does not modify existing comparison files. The 2024 validation
season is used to identify the best candidate; 2025/2026 are descriptive only.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ORIGINAL_PRIOR = {'batter': 1200.0, 'pitcher': 1080.0}
EXPECTED_N = {('batter', 2025): 106, ('pitcher', 2025): 86,
              ('batter', 2026): 109, ('pitcher', 2026): 83}
REQUIRED = {'PlayerId', 'actual', 'league_mean_prediction',
            'marcel_pre_age', 'marcel_adapted_prediction',
            'projected_age', 'historical_weighted_volume'}


def age_factor(ages):
    ages = np.asarray(ages, dtype=float)
    result = np.ones_like(ages)
    valid = np.isfinite(ages)
    younger = valid & (ages <= 29)
    older = valid & (ages > 29)
    result[younger] += 0.006 * (29 - ages[younger])
    result[older] += 0.003 * (29 - ages[older])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--input-dir', type=Path, default=base / 'results' / 'baseline_comparison')
    parser.add_argument('--output-dir', type=Path, default=base / 'results' / 'marcel_sensitivity_results')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = []
    validation_winners = []
    for kind in ('batter', 'pitcher'):
        prior0 = ORIGINAL_PRIOR[kind]
        # Sensitivity choices are fixed in advance; not optimized on test seasons.
        candidates = [0.5 * prior0, prior0, 2.0 * prior0]
        for year in (2024, 2025, 2026):
            path = args.input_dir / f'{kind}_{year}_comparison.csv'
            if not path.is_file():
                raise FileNotFoundError(f'파일을 찾을 수 없습니다: {path}')
            df = pd.read_csv(path, dtype={'PlayerId': str})
            missing = REQUIRED - set(df.columns)
            if missing:
                raise ValueError(f'{path.name}에 필요한 열이 없습니다: {sorted(missing)}')
            if df['PlayerId'].isna().any() or df['PlayerId'].duplicated().any():
                raise ValueError(f'{path.name}: PlayerId 결측 또는 중복')
            if (kind, year) in EXPECTED_N and len(df) != EXPECTED_N[(kind, year)]:
                raise ValueError(f'{path.name}: 선수 수 {len(df)}명, 예상 {EXPECTED_N[(kind, year)]}명')

            cols = ['actual', 'league_mean_prediction', 'marcel_pre_age',
                    'marcel_adapted_prediction', 'projected_age',
                    'historical_weighted_volume']
            numeric = df[cols].apply(pd.to_numeric, errors='coerce')
            for col in cols:
                if col == 'projected_age':
                    continue  # Missing age means no age correction in original code.
                if not np.isfinite(numeric[col].to_numpy(float)).all():
                    raise ValueError(f'{path.name}: {col}에 비정상 값이 있습니다.')
            y = numeric['actual'].to_numpy(float)
            mu = numeric['league_mean_prediction'].to_numpy(float)
            pre = numeric['marcel_pre_age'].to_numpy(float)
            stored = numeric['marcel_adapted_prediction'].to_numpy(float)
            v = numeric['historical_weighted_volume'].to_numpy(float)
            ages = numeric['projected_age'].to_numpy(float)
            if (v < 0).any():
                raise ValueError(f'{path.name}: 음수 출전량')

            # Reconstruct the original weighted numerator without reloading raw data.
            weighted_sum = pre * (v + prior0) - prior0 * mu
            factor = age_factor(ages)
            original_recreated = mu + (pre - mu) * factor
            discrepancy = np.max(np.abs(original_recreated - stored))
            if discrepancy > 1e-6:
                raise ValueError(
                    f'{path.name}: 원래 예측값 재현 불일치 (최대 차이 {discrepancy:.8g}). '
                    '입력 CSV 또는 기존 계산식을 확인하세요.'
                )

            for prior in candidates:
                raw = (weighted_sum + prior * mu) / (v + prior)
                for use_age in (False, True):
                    predicted = mu + (raw - mu) * (factor if use_age else 1.0)
                    mae = float(np.mean(np.abs(y - predicted)))
                    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
                    name = f'P={prior:g}, age={"on" if use_age else "off"}'
                    rows.append({'kind': kind, 'year': year,
                                 'split': 'validation' if year == 2024 else ('test' if year == 2025 else 'additional_evaluation'),
                                 'candidate': name, 'prior': prior,
                                 'age_adjustment': 'on' if use_age else 'off',
                                 'n': len(df), 'MAE': mae, 'RMSE': rmse})
                    predictions.append(pd.DataFrame({
                        'kind': kind, 'year': year, 'PlayerId': df['PlayerId'],
                        'candidate': name, 'actual': y, 'prediction': predicted
                    }))
            print(f'[완료] {path.name}: {len(df)}명, 기존 Marcel 예측값 재현 확인')

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / 'sensitivity_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / 'sensitivity_predictions.csv', index=False, encoding='utf-8-sig')

    # Choose based ONLY on validation MAE; ties resolved by lower complexity:
    # no age correction, then original-prior proximity.
    for kind in ('batter', 'pitcher'):
        valid = result[(result.kind == kind) & (result.year == 2024)].copy()
        valid['age_tiebreak'] = (valid.age_adjustment == 'on').astype(int)
        valid['prior_tiebreak'] = abs(np.log(valid.prior / ORIGINAL_PRIOR[kind]))
        best = valid.sort_values(['MAE', 'age_tiebreak', 'prior_tiebreak']).iloc[0]
        validation_winners.append({'kind': kind, 'selected_from': '2024 validation only',
                                   'candidate': best.candidate, 'prior': best.prior,
                                   'age_adjustment': best.age_adjustment,
                                   'validation_MAE': best.MAE})
    winners = pd.DataFrame(validation_winners)
    winners.to_csv(args.output_dir / 'validation_selected_candidates.csv',
                   index=False, encoding='utf-8-sig')

    print('\n[2024년 검증 기준 최저 MAE 설정 — 탐색적 분석]')
    print(winners.to_string(index=False))
    print('\n[원래 설정(P=기존값, 연령 보정 ON)과 연령 보정 OFF 비교]')
    original = result[(result.kind == 'batter') & (result.prior == ORIGINAL_PRIOR['batter']) |
                      (result.kind == 'pitcher') & (result.prior == ORIGINAL_PRIOR['pitcher'])]
    print(original[['kind','year','age_adjustment','MAE','RMSE']].to_string(index=False))
    print('\n저장 위치:', args.output_dir.resolve())
    print('주의: 2025/2026 결과를 근거로 설정을 다시 선택하면 독립 평가가 훼손됩니다.')
    print('이 결과는 기존 모델을 자동으로 변경하지 않습니다.')


if __name__ == '__main__':
    main()

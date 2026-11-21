"""KBO 선수별 예측값을 이용한 쌍대 부트스트랩 MAE 차이 분석.

실행: python analyze_significance.py --input-dir baseline_comparison --output-dir significance_results
필요 패키지: pandas numpy
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

MODEL_COLUMNS = {
    'Ridge': 'ridge_prediction',
    'XGBoost': 'xgboost_prediction',
    'Marcel-inspired': 'marcel_adapted_prediction',
    'League mean': 'league_mean_prediction',
}
PAIRS = [('Ridge', 'XGBoost'), ('Ridge', 'Marcel-inspired'), ('Ridge', 'League mean')]


def compare_pair(actual, pred_a, pred_b, rng, n_boot):
    """차이 = MAE(A) - MAE(B); 음수이면 A가 더 좋음."""
    err_a = np.abs(pred_a - actual)
    err_b = np.abs(pred_b - actual)
    differences = err_a - err_b
    n = len(differences)
    observed = float(differences.mean())
    # 선수 단위로 같은 인덱스를 뽑으므로 쌍대 구조가 유지됨.
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = differences[idx].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path('significance_results'))
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error('--bootstrap은 1000 이상을 권장합니다.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = []

    for kind in ('batter', 'pitcher'):
        for year in (2025, 2026):
            path = args.input_dir / f'{kind}_{year}_comparison.csv'
            if not path.is_file():
                raise FileNotFoundError(f'파일이 없습니다: {path}')
            df = pd.read_csv(path)
            required = {'PlayerId', 'actual', 'ridge_prediction',
                        'marcel_adapted_prediction', 'league_mean_prediction'}
            if year == 2025:
                required.add('xgboost_prediction')
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f'{path.name}에 열이 없습니다: {sorted(missing)}')
            if df['PlayerId'].isna().any() or df['PlayerId'].duplicated().any():
                raise ValueError(f'{path.name}: PlayerId 결측 또는 중복이 있습니다.')
            if len(df) != ({('batter', 2025):106, ('pitcher', 2025):86,
                           ('batter', 2026):109, ('pitcher', 2026):83}[(kind,year)]):
                raise ValueError(f'{path.name}: 예상 평가 선수 수와 다릅니다 ({len(df)}).')

            pairs = PAIRS if year == 2025 else PAIRS[1:]
            for a, b in pairs:
                columns = ['actual', MODEL_COLUMNS[a], MODEL_COLUMNS[b]]
                values = df[columns].apply(pd.to_numeric, errors='coerce')
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f'{path.name}: {a} vs {b}에 결측/비정상 수치가 있습니다.')
                y, pa, pb = (values[c].to_numpy(dtype=float) for c in columns)
                delta, low, high = compare_pair(y, pa, pb, rng, args.bootstrap)
                results.append({
                    'kind': kind, 'year': year, 'n': len(df),
                    'model_A': a, 'model_B': b,
                    'MAE_A': np.mean(np.abs(pa-y)),
                    'MAE_B': np.mean(np.abs(pb-y)),
                    'MAE_A_minus_B': delta,
                    'CI95_lower': low, 'CI95_upper': high,
                    'CI_contains_zero': bool(low <= 0 <= high),
                    'bootstrap_iterations': args.bootstrap, 'seed': args.seed,
                })
            print(f'[완료] {path.name}: {len(df)}명, {len(pairs)}개 비교')

    output = args.output_dir / 'paired_bootstrap_mae.csv'
    pd.DataFrame(results).to_csv(output, index=False, encoding='utf-8-sig')
    print(f'\n저장 완료: {output.resolve()}')
    print('해석: MAE_A_minus_B < 0이면 A가 더 낮은 MAE입니다.')
    print('주의: CI에 0이 포함되지 않아도 다중비교/표본 선택의 한계를 고려하세요.')


if __name__ == '__main__':
    main()

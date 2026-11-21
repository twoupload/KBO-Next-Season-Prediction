"""KBO next-season prediction v3: lag 1/2/3, native/common cohorts.

Install: python -m pip install pandas numpy scikit-learn xgboost
Run:     python train_model_v3.py
Quick:   python train_model_v3.py --models baseline ridge
Options: python train_model_v3.py --cohorts common --models baseline ridge random_forest xgboost

Inputs: kbo_linked_2015_2025/{batter,pitcher}_lag{1,2,3}_qualified.csv
Outputs: ml_results_v3/ (does not modify prior experiments)

IMPORTANT: qualified inputs select on next-year playing time. Results are
conditional on the player satisfying that future participation threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results' / 'ml_results_v3'
TRAIN_END, VALID_YEAR, TEST_YEAR = 2023, 2024, 2025
SEED = 42
KINDS = {
    'batter': ('Target_wRC+', 'wRC+_lag1'),
    'pitcher': ('Target_FIP', 'FIP_lag1'),
}
# Modest, reproducible grids. All selections are made on 2024 MAE only.
GRIDS = {
    'ridge': [{'alpha': a} for a in (0.1, 1., 10., 100., 1000., 3000., 10000.)],
    'random_forest': [
        {'n_estimators': 200, 'max_depth': depth, 'min_samples_leaf': leaf}
        for depth in (4, 8) for leaf in (2, 5)
    ],
    'xgboost': [
        {'n_estimators': n, 'max_depth': depth, 'learning_rate': 0.05,
         'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 5.0}
        for n in (150, 300) for depth in (2, 3)
    ],
}


def locate(kind: str, lag: int) -> Path:
    name = f'{kind}_lag{lag}_qualified.csv'
    candidates = [ROOT / folder / name for folder in
                  ('data/kbo_linked_2015_2025', 'kbo_linked_2015_2025', 'output', 'data', '.')]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'{name} 파일을 찾을 수 없습니다. 확인 위치:\n' +
                            '\n'.join(str(p) for p in candidates))


def read_dataset(kind: str, lag: int):
    target, baseline = KINDS[kind]
    path = locate(kind, lag)
    df = pd.read_csv(path, low_memory=False, dtype={'PlayerId': str})
    required = ['PlayerId', 'TargetSeason', target, baseline]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: 필수 열 누락: {missing}')
    for col in ('TargetSeason', target, baseline):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[required].isna().any().any():
        raise ValueError(f'{path}: 선수ID, 연도, 정답, Baseline 중 결측값 발견')
    if df.duplicated(['PlayerId', 'TargetSeason']).any():
        raise ValueError(f'{path}: 선수ID+예측시즌 중복 발견')
    # Only lagged numeric statistics may enter the model. Never target-year info.
    suffixes = tuple(f'_lag{i}' for i in range(1, lag + 1))
    features = [c for c in df.columns if c.endswith(suffixes)
                and not c.startswith('InputQualified_')
                and pd.to_numeric(df[c], errors='coerce').notna().any()]
    if baseline not in features:
        raise ValueError(f'{path}: Baseline 지표가 입력 변수에서 누락')
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    # Key as tuple avoids accidental concatenation collisions.
    df['_key'] = list(zip(df['PlayerId'], df['TargetSeason'].astype(int)))
    return df, features, path


def score(actual, predicted):
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': float(r2_score(actual, predicted))}


def model_for(name, params):
    if name == 'ridge':
        estimator = Ridge(**params)
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()), ('model', estimator)
        ])
    if name == 'random_forest':
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    elif name == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError('XGBoost 설치: python -m pip install xgboost') from exc
        estimator = XGBRegressor(objective='reg:squarederror', random_state=SEED,
                                 n_jobs=4, tree_method='hist', **params)
    else:
        raise ValueError(name)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', estimator)
    ])


def evaluate(kind, lag, cohort, df, features, path, models):
    target, baseline = KINDS[kind]
    train = df[df.TargetSeason <= TRAIN_END].copy()
    valid = df[df.TargetSeason == VALID_YEAR].copy()
    test = df[df.TargetSeason == TEST_YEAR].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f'{kind} lag{lag} {cohort}: 빈 학습/검증/테스트 분할')
    xtr, ytr = train[features], train[target]
    xva, yva = valid[features], valid[target]
    combined = pd.concat([train, valid], ignore_index=True)
    valid_preds = {'baseline': valid[baseline].to_numpy()}
    test_preds = {'baseline': test[baseline].to_numpy()}
    tuning, chosen = [], {}
    for name in models:
        if name == 'baseline':
            continue
        best_mae, best_params, best_pred = float('inf'), None, None
        for params in GRIDS[name]:
            estimator = model_for(name, params)
            estimator.fit(xtr, ytr)
            pred = estimator.predict(xva)
            metrics = score(yva, pred)
            tuning.append({'type': kind, 'cohort': cohort, 'lag': lag,
                           'model': name, 'parameters': json.dumps(params), **metrics})
            if metrics['MAE'] < best_mae:
                best_mae, best_params, best_pred = metrics['MAE'], params.copy(), pred
        chosen[name] = best_params
        valid_preds[name] = best_pred
        final_model = model_for(name, best_params)
        final_model.fit(combined[features], combined[target])
        test_preds[name] = final_model.predict(test[features])
        print(f'  {name}: 2024 MAE={best_mae:.4f}, 설정={best_params}', flush=True)
    stem = f'{kind}_{cohort}_lag{lag}'
    pd.DataFrame(tuning).to_csv(OUTPUT / f'{stem}_tuning.csv', index=False, encoding='utf-8-sig')
    rows = []
    for split, frame, predictions in [('validation', valid, valid_preds), ('test', test, test_preds)]:
        cols = ['PlayerId', 'TargetSeason'] + [c for c in ('TargetName', 'TargetTeam') if c in frame]
        result = frame[cols].copy()
        result['actual'] = frame[target].to_numpy()
        for name, values in predictions.items():
            result[f'{name}_prediction'] = values
            result[f'{name}_abs_error'] = np.abs(result['actual'].to_numpy() - values)
            rows.append({'type': kind, 'cohort': cohort, 'lag': lag, 'split': split,
                         'model': name, 'n': len(frame), **score(frame[target], values)})
        result.to_csv(OUTPUT / f'{stem}_{split}_predictions.csv', index=False,
                      encoding='utf-8-sig')
    info = {'type': kind, 'cohort': cohort, 'lag': lag, 'input_file': str(path),
            'features': features, 'train_count': len(train), 'validation_count': len(valid),
            'test_count': len(test), 'train_end_year': TRAIN_END,
            'validation_year': VALID_YEAR, 'test_year': TEST_YEAR,
            'selected_by': '2024 validation MAE only',
            'final_fit': 'train through 2024 after selection',
            'chosen_hyperparameters': chosen, 'seed': SEED,
            'limitation': 'qualified dataset conditions on future playing time'}
    (OUTPUT / f'{stem}_experiment_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description='KBO lag 1/2/3 native vs common cohort experiments')
    parser.add_argument('--cohorts', choices=['both', 'native', 'common'], default='both')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'ridge', 'random_forest', 'xgboost'],
                        default=['baseline', 'ridge', 'random_forest', 'xgboost'])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows, counts = [], []
    for kind in KINDS:
        loaded = {lag: read_dataset(kind, lag) for lag in (1, 2, 3)}
        # Same player-season observations in all three lag files.
        common_keys = set.intersection(*(set(loaded[lag][0]['_key']) for lag in (1, 2, 3)))
        # Assert that the shared outcome is identical across all files.
        target = KINDS[kind][0]
        shared = [loaded[lag][0].set_index('_key').loc[sorted(common_keys), target].to_numpy()
                  for lag in (1, 2, 3)]
        if not all(np.allclose(shared[0], s, equal_nan=True) for s in shared[1:]):
            raise ValueError(f'{kind}: 공통 표본에서 정답값 불일치; 연결 CSV 확인 필요')
        for cohort in (['native', 'common'] if args.cohorts == 'both' else [args.cohorts]):
            for lag in (1, 2, 3):
                original, features, path = loaded[lag]
                frame = original if cohort == 'native' else original[original['_key'].isin(common_keys)].copy()
                print(f'\n[{kind} / {cohort} / lag{lag}] 전체 {len(frame)}건 / 입력변수 {len(features)}개',
                      flush=True)
                result = evaluate(kind, lag, cohort, frame, features, path, args.models)
                all_rows.extend(result)
                counts.append({'type': kind, 'cohort': cohort, 'lag': lag,
                               'train_n': int((frame.TargetSeason <= TRAIN_END).sum()),
                               'validation_n': int((frame.TargetSeason == VALID_YEAR).sum()),
                               'test_n': int((frame.TargetSeason == TEST_YEAR).sum())})
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUTPUT / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(counts).to_csv(OUTPUT / 'sample_counts.csv', index=False, encoding='utf-8-sig')
    print('\n=== 2024 validation metrics (model selection) ===')
    print(metrics[metrics.split == 'validation'].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== 2025 test metrics (reporting only; NEVER select models using these) ===')
    print(metrics[metrics.split == 'test'].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n완료: {OUTPUT}')


if __name__ == '__main__':
    main()

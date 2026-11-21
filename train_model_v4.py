"""KBO 투수 보직 변수 제거 실험(v4): FIP 다음 시즌 예측.

설치: python -m pip install pandas numpy scikit-learn xgboost
실행: python train_model_v4.py
옵션: python train_model_v4.py --lags 1 2 3 --cohort common
빠른 점검: python train_model_v4.py --models baseline ridge

A(all): 기존 전체 변수
B(no_gs_rate): GS_rate_lagN 제외
C(no_gs_and_rate): GS_lagN 및 GS_rate_lagN 제외

동일 lag/cohort 내에서 세 조건은 동일 선수·시즌, 동일 분할을 사용합니다.
2024 검증 MAE로만 각 조건의 하이퍼파라미터를 선택합니다.
2025 테스트는 보고 전용입니다. 기존 결과 파일은 수정하지 않습니다.
주의: qualified 자료는 미래 시즌 출전량을 조건으로 필터링했습니다.
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
OUTPUT = ROOT / 'ml_results_v4'
TRAIN_END, VALID_YEAR, TEST_YEAR = 2023, 2024, 2025
SEED = 42
TARGET, BASELINE = 'Target_FIP', 'FIP_lag1'
VARIANTS = ('all', 'no_gs_rate', 'no_gs_and_rate')
GRIDS = {
    'ridge': [{'alpha': a} for a in (0.1, 1., 10., 100., 1000., 3000., 10000.)],
    'random_forest': [
        {'n_estimators': 200, 'max_depth': d, 'min_samples_leaf': leaf}
        for d in (4, 8) for leaf in (2, 5)
    ],
    'xgboost': [
        {'n_estimators': n, 'max_depth': d, 'learning_rate': 0.05,
         'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 5.0}
        for n in (150, 300) for d in (2, 3)
    ],
}


def locate(lag):
    name = f'pitcher_lag{lag}_qualified.csv'
    candidates = [ROOT / folder / name for folder in
                  ('kbo_linked_2015_2025', 'output', 'data', '.')]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError('파일을 찾을 수 없습니다: ' + name + '\n' + '\n'.join(map(str, candidates)))


def load(lag):
    path = locate(lag)
    df = pd.read_csv(path, dtype={'PlayerId': str}, low_memory=False)
    required = ['PlayerId', 'TargetSeason', TARGET, BASELINE]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'{path}: 필수 열 누락 {missing}')
    for col in ('TargetSeason', TARGET, BASELINE):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[required].isna().any().any():
        raise ValueError(f'{path}: ID/연도/정답/베이스라인 결측 발견')
    if df.duplicated(['PlayerId', 'TargetSeason']).any():
        raise ValueError(f'{path}: 선수ID+시즌 중복 발견')
    suffixes = tuple(f'_lag{i}' for i in range(1, lag + 1))
    features = [c for c in df if c.endswith(suffixes)
                and not c.startswith('InputQualified_')
                and pd.to_numeric(df[c], errors='coerce').notna().any()]
    for i in range(1, lag + 1):
        for col in (f'GS_lag{i}', f'GS_rate_lag{i}'):
            if col not in features:
                raise ValueError(f'{path}: 보직 변수 {col} 누락. 원본 CSV를 확인하세요.')
    if BASELINE not in features:
        raise ValueError(f'{path}: {BASELINE} 누락')
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df['_key'] = list(zip(df.PlayerId, df.TargetSeason.astype(int)))
    return df, features, path


def subset_features(all_features, variant):
    if variant == 'all':
        return all_features.copy()
    if variant == 'no_gs_rate':
        return [c for c in all_features if not c.startswith('GS_rate_lag')]
    return [c for c in all_features
            if not (c.startswith('GS_rate_lag') or c.startswith('GS_lag'))]


def score(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred))}


def make_model(name, params):
    if name == 'ridge':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()), ('model', Ridge(**params))])
    if name == 'random_forest':
        regressor = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    elif name == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError('설치 필요: python -m pip install xgboost') from exc
        regressor = XGBRegressor(objective='reg:squarederror', random_state=SEED,
                                 n_jobs=4, tree_method='hist', **params)
    else:
        raise ValueError(name)
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('model', regressor)])


def evaluate(lag, cohort, variant, frame, features, path, models):
    train = frame[frame.TargetSeason <= TRAIN_END].copy()
    valid = frame[frame.TargetSeason == VALID_YEAR].copy()
    test = frame[frame.TargetSeason == TEST_YEAR].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f'lag{lag}/{cohort}: 학습·검증·테스트 중 빈 분할')
    both = pd.concat([train, valid], ignore_index=True)
    preds = {
        'validation': {'baseline': valid[BASELINE].to_numpy()},
        'test': {'baseline': test[BASELINE].to_numpy()}
    }
    chosen, tuning_rows = {}, []
    for model in models:
        if model == 'baseline':
            continue
        best = (float('inf'), None, None)
        for params in GRIDS[model]:
            est = make_model(model, params)
            est.fit(train[features], train[TARGET])
            prediction = est.predict(valid[features])
            metrics = score(valid[TARGET], prediction)
            tuning_rows.append({'cohort': cohort, 'lag': lag, 'variant': variant,
                                'model': model, 'parameters': json.dumps(params), **metrics})
            if metrics['MAE'] < best[0]:
                best = (metrics['MAE'], params.copy(), prediction)
        chosen[model] = best[1]
        preds['validation'][model] = best[2]
        final_est = make_model(model, best[1])
        final_est.fit(both[features], both[TARGET])
        preds['test'][model] = final_est.predict(test[features])
        print(f'    {model}: 2024 검증 MAE={best[0]:.4f}, params={best[1]}', flush=True)

    stem = f'pitcher_{cohort}_lag{lag}_{variant}'
    pd.DataFrame(tuning_rows).to_csv(OUTPUT / f'{stem}_tuning.csv', index=False, encoding='utf-8-sig')
    rows = []
    for split, part in (('validation', valid), ('test', test)):
        cols = ['PlayerId', 'TargetSeason'] + [c for c in ('TargetName', 'TargetTeam') if c in part]
        result = part[cols].copy()
        result['actual'] = part[TARGET].to_numpy()
        for model, prediction in preds[split].items():
            result[f'{model}_prediction'] = prediction
            result[f'{model}_abs_error'] = np.abs(part[TARGET].to_numpy() - prediction)
            rows.append({'type': 'pitcher', 'cohort': cohort, 'lag': lag,
                         'variant': variant, 'split': split, 'model': model,
                         'n': len(part), 'n_features': len(features),
                         **score(part[TARGET], prediction)})
        result.to_csv(OUTPUT / f'{stem}_{split}_predictions.csv', index=False,
                      encoding='utf-8-sig')
    info = {'type': 'pitcher', 'cohort': cohort, 'lag': lag, 'variant': variant,
            'source': str(path), 'features': features, 'removed_features':
            [f'GS_rate_lag{i}' for i in range(1, lag + 1)] if variant == 'no_gs_rate' else
            ([c for i in range(1, lag + 1) for c in (f'GS_lag{i}', f'GS_rate_lag{i}')]
             if variant == 'no_gs_and_rate' else []),
            'train_n': len(train), 'validation_n': len(valid), 'test_n': len(test),
            'train_end_year': TRAIN_END, 'validation_year': VALID_YEAR,
            'test_year': TEST_YEAR, 'seed': SEED,
            'chosen_hyperparameters': chosen,
            'selection': '2024 validation MAE only',
            'final_fit': '2016-2024 train+validation after tuning',
            'warning': 'qualified data conditions on future playing time; results are conditional'}
    (OUTPUT / f'{stem}_experiment_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


def make_comparison(metrics):
    # Same split, model, lag and cohort, hence a paired descriptive comparison.
    key = ['cohort', 'lag', 'split', 'model']
    pivot = metrics.pivot(index=key, columns='variant', values='MAE')
    required = set(VARIANTS)
    if not required.issubset(pivot.columns):
        raise ValueError('A/B/C 실험 결과 일부 누락')
    out = pivot.reset_index()
    out['delta_B_minus_A'] = out['no_gs_rate'] - out['all']
    out['delta_C_minus_A'] = out['no_gs_and_rate'] - out['all']
    out['best_variant_by_MAE'] = out[list(VARIANTS)].idxmin(axis=1)
    out.to_csv(OUTPUT / 'role_ablation_comparison.csv', index=False, encoding='utf-8-sig')
    return out


def main():
    parser = argparse.ArgumentParser(description='KBO 투수 선발/불펜 관련 변수 ablation')
    parser.add_argument('--lags', nargs='+', type=int, choices=[1, 2, 3], default=[1],
                        help='기본값: 1. 1 2 3으로 지정하면 모두 실험')
    parser.add_argument('--cohort', choices=['native', 'common'], default='native',
                        help='common은 lag1~3 모두 있는 동일 선수-시즌만 사용')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'ridge', 'random_forest', 'xgboost'],
                        default=['baseline', 'ridge', 'random_forest', 'xgboost'])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {lag: load(lag) for lag in (1, 2, 3) if args.cohort == 'common' or lag in args.lags}
    common_keys = None
    if args.cohort == 'common':
        common_keys = set.intersection(*(set(data[lag][0]['_key']) for lag in (1, 2, 3)))
        if not common_keys:
            raise ValueError('lag1~3 공통 선수-시즌이 없습니다.')
        reference = data[1][0].set_index('_key').loc[sorted(common_keys), TARGET].to_numpy()
        for lag in (2, 3):
            current = data[lag][0].set_index('_key').loc[sorted(common_keys), TARGET].to_numpy()
            if not np.allclose(reference, current):
                raise ValueError('lag별 공통 선수-시즌 정답이 다릅니다.')
    all_rows = []
    for lag in args.lags:
        frame, all_features, path = data[lag]
        if common_keys is not None:
            frame = frame[frame['_key'].isin(common_keys)].copy()
        for variant in VARIANTS:
            features = subset_features(all_features, variant)
            print(f'\n[투수 / {args.cohort} / lag{lag} / {variant}] '
                  f'표본={len(frame)}, 변수={len(features)}', flush=True)
            all_rows.extend(evaluate(lag, args.cohort, variant, frame, features, path, args.models))
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUTPUT / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    compare = make_comparison(metrics)
    print('\n=== 2024 검증: 보직 변수 제거에 따른 MAE 차이 (음수=제거 후 개선) ===')
    print(compare[compare.split == 'validation'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n=== 2025 테스트: 보고 전용 (설정 선택에 사용 금지) ===')
    print(compare[compare.split == 'test'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n완료: {OUTPUT}')


if __name__ == '__main__':
    main()

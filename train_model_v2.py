"""KBO 다음 시즌 예측 v2: Baseline / Ridge / Random Forest / XGBoost.

실행:
  python -m pip install pandas numpy scikit-learn xgboost
  python train_model_v2.py

입력: batter_lag1_qualified.csv, pitcher_lag1_qualified.csv
출력: ml_results_v2/ (기존 ml_results/는 건드리지 않음)
연도: 학습 2023년까지, 검증 2024년, 최종 테스트 2025년
주의: qualified 자료는 정답 시즌 출전량으로 표본을 제한한 조건부 예측입니다.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'ml_results_v2'
TRAIN_END_YEAR = 2023
VALID_YEAR = 2024
TEST_YEAR = 2025
RANDOM_STATE = 42

CONFIG = {
    'batter': {'filename': 'batter_lag1_qualified.csv', 'target': 'Target_wRC+', 'baseline': 'wRC+_lag1'},
    'pitcher': {'filename': 'pitcher_lag1_qualified.csv', 'target': 'Target_FIP', 'baseline': 'FIP_lag1'},
}

# 2024년 검증 MAE로만 설정을 선택합니다. 2025년은 선택에 사용하지 않습니다.
GRIDS = {
    'ridge': [{'alpha': a} for a in [0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0]],
    'random_forest': [
        {'n_estimators': 250, 'max_depth': depth, 'min_samples_leaf': leaf}
        for depth in [4, 8, None] for leaf in [2, 5]
    ],
    'xgboost': [
        {'n_estimators': n, 'max_depth': depth, 'learning_rate': lr,
         'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 5.0}
        for n in [150, 350] for depth in [2, 3] for lr in [0.03, 0.07]
    ],
}


def find_input_file(filename):
    locations = [ROOT / folder / filename for folder in
                 ['data/kbo_linked_2015_2025', 'kbo_linked_2015_2025', 'output', 'data', '.']]
    for path in locations:
        if path.is_file():
            return path
    raise FileNotFoundError(f'{filename} 파일을 찾지 못했습니다. 확인한 경로:\n' +
                            '\n'.join(str(p) for p in locations))


def score(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred))}


def make_model(name, params):
    if name == 'ridge':
        regressor = Ridge(**params)
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
            ('model', regressor),
        ])
    if name == 'random_forest':
        regressor = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    elif name == 'xgboost':
        regressor = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE,
                                 n_jobs=4, tree_method='hist', **params)
    else:
        raise ValueError(f'알 수 없는 모델: {name}')
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', regressor),
    ])


def load_data(kind, cfg):
    path = find_input_file(cfg['filename'])
    df = pd.read_csv(path, low_memory=False)
    required = ['PlayerId', 'TargetSeason', cfg['target'], cfg['baseline']]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name}: 필요한 열이 없습니다: {missing}')
    for c in ['TargetSeason', cfg['target'], cfg['baseline']]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[required].isna().any().any():
        raise ValueError(f'{path.name}: 필수 열에 결측값이 있습니다. 원본을 확인하세요.')
    if df.duplicated(['PlayerId', 'TargetSeason']).any():
        raise ValueError(f'{path.name}: 선수-시즌 중복이 있습니다.')
    features = [c for c in df.columns if c.endswith('_lag1') and c != 'InputQualified_lag1'
                and pd.to_numeric(df[c], errors='coerce').notna().any()]
    if cfg['baseline'] not in features:
        raise ValueError('전년도 성적 변수가 입력 변수에 없습니다.')
    # 입력 변수는 과거 시즌(_lag1)만 사용합니다. 정답 시즌 정보 사용 금지.
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    train = df[df.TargetSeason <= TRAIN_END_YEAR].copy()
    valid = df[df.TargetSeason == VALID_YEAR].copy()
    test = df[df.TargetSeason == TEST_YEAR].copy()
    if not all(len(x) for x in [train, valid, test]):
        raise ValueError(f'{kind}: 학습/검증/테스트 중 빈 데이터가 있습니다.')
    return path, features, train, valid, test


def predictions_csv(kind, split, frame, cfg, preds):
    cols = ['PlayerId', 'TargetSeason']
    cols += [c for c in ['TargetName', 'TargetTeam'] if c in frame.columns]
    out = frame[cols].copy()
    out['actual'] = frame[cfg['target']].to_numpy()
    for name, values in preds.items():
        out[f'{name}_prediction'] = values
        out[f'{name}_abs_error'] = np.abs(out['actual'].to_numpy() - values)
    out.to_csv(RESULT_DIR / f'{kind}_{split}_predictions.csv', index=False, encoding='utf-8-sig')


def run_kind(kind, cfg):
    path, features, train, valid, test = load_data(kind, cfg)
    Xtr, ytr = train[features], train[cfg['target']]
    Xva, yva = valid[features], valid[cfg['target']]
    Xte, yte = test[features], test[cfg['target']]
    combined = pd.concat([train, valid], ignore_index=True)
    Xcombined, ycombined = combined[features], combined[cfg['target']]

    valid_preds = {'baseline': valid[cfg['baseline']].to_numpy()}
    test_preds = {'baseline': test[cfg['baseline']].to_numpy()}
    chosen = {}
    tuning_rows = []
    print(f'\n[{kind}] 학습 {len(train)} / 검증 {len(valid)} / 테스트 {len(test)}; 입력변수 {len(features)}개')

    for name, grid in GRIDS.items():
        best_params, best_mae, best_pred = None, float('inf'), None
        for params in grid:
            model = make_model(name, params)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            metrics = score(yva, pred)
            tuning_rows.append({'model': name, 'parameters': json.dumps(params, ensure_ascii=False), **metrics})
            if metrics['MAE'] < best_mae:
                best_mae, best_params, best_pred = metrics['MAE'], params.copy(), pred
        chosen[name] = best_params
        valid_preds[name] = best_pred
        final_model = make_model(name, best_params)
        final_model.fit(Xcombined, ycombined)
        test_preds[name] = final_model.predict(Xte)
        print(f'  {name}: 검증 MAE={best_mae:.4f}, 설정={best_params}')

    pd.DataFrame(tuning_rows).to_csv(RESULT_DIR / f'{kind}_hyperparameter_validation.csv',
                                      index=False, encoding='utf-8-sig')
    predictions_csv(kind, 'validation', valid, cfg, valid_preds)
    predictions_csv(kind, 'test', test, cfg, test_preds)
    rows = []
    for split, frame, preds in [('validation', valid, valid_preds), ('test', test, test_preds)]:
        for name, pred in preds.items():
            rows.append({'type': kind, 'split': split, 'model': name,
                         'n': len(frame), **score(frame[cfg['target']], pred)})
    info = {'type': kind, 'input_file': str(path), 'target': cfg['target'],
            'baseline': cfg['baseline'], 'features': features,
            'train_end_year': TRAIN_END_YEAR, 'validation_year': VALID_YEAR,
            'test_year': TEST_YEAR, 'train_count': len(train),
            'validation_count': len(valid), 'test_count': len(test),
            'random_state': RANDOM_STATE, 'chosen_hyperparameters': chosen,
            'model_selection': '2024 validation MAE only',
            'final_fit': 'train through 2024 after parameter selection',
            'caution': 'qualified dataset filters on target-year playing time; conditional prediction'}
    (RESULT_DIR / f'{kind}_experiment_info.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    return rows


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, cfg in CONFIG.items():
        rows.extend(run_kind(kind, cfg))
    report = pd.DataFrame(rows)
    report.to_csv(RESULT_DIR / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    print('\n=== 성능 비교 (MAE/RMSE 낮을수록, R2 높을수록 좋음) ===')
    print(report.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n완료: {RESULT_DIR}')


if __name__ == '__main__':
    main()

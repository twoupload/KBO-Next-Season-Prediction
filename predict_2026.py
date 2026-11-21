"""KBO 2026 season forecasts from 2015-2025 data ONLY.

Run in KBO_Prediction folder:
    python -m pip install pandas numpy scikit-learn
    python predict_2026.py

Inputs:
  data/kbo_batter_2015_2025_active.csv (or project root)
  data/kbo_pitcher_2015_2025_active.csv (or project root)
  kbo_linked_2015_2025/batter_lag1_qualified.csv
  kbo_linked_2015_2025/pitcher_lag1_qualified.csv
  Optional: ml_results_v2/{batter,pitcher}_experiment_info.json

Outputs: predictions_2026/{batter,pitcher}_2026_predictions.csv,
         predictions_2026/prediction_summary.csv,
         predictions_2026/prediction_manifest.json

No 2026 actual-results files are read by this program.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'predictions_2026'

# Locked from the v2 2024 validation run; NEVER tune using 2026 outcomes.
CONFIG = {
    'batter': {'target': 'Target_wRC+', 'metric': 'wRC+', 'volume': 'PA',
               'alpha': 0.1, 'min_volume': 100},
    'pitcher': {'target': 'Target_FIP', 'metric': 'FIP', 'volume': 'IP_outs',
                'alpha': 1000.0, 'min_volume': 90},
}


def find_file(name, folders):
    for folder in folders:
        path = ROOT / folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'{name} 파일이 없습니다. 확인한 폴더: {folders}')


def ip_to_outs(series):
    """KBO IP notation: 12.1=12 innings + 1 out, 12.2=12 innings + 2 outs."""
    result = []
    for val in series:
        if pd.isna(val):
            result.append(np.nan)
            continue
        raw = str(val).strip()
        match = re.fullmatch(r'(\d+)(?:\.(\d+))?', raw)
        if not match:
            raise ValueError(f'잘못된 IP 값: {val!r}')
        innings = int(match.group(1))
        fraction = match.group(2) or '0'
        if fraction not in ('0', '00', '1', '10', '2', '20'):
            raise ValueError(f'IP는 .0, .1, .2만 허용합니다: {val!r}')
        result.append(innings * 3 + int(fraction[0]))
    return pd.Series(result, index=series.index, dtype='float64')


def check_ids(frame, label, year_col):
    if frame['PlayerId'].isna().any():
        raise ValueError(f'{label}: PlayerId 결측')
    if frame.duplicated(['PlayerId', year_col]).any():
        raise ValueError(f'{label}: 선수/시즌 중복')


def run(kind, cfg):
    linked_path = find_file(f'{kind}_lag1_qualified.csv',
                            ['kbo_linked_2015_2025', 'data', '.'])
    source_path = find_file(f'kbo_{kind}_2015_2025_active.csv', ['data', '.'])
    linked = pd.read_csv(linked_path, low_memory=False, dtype={'PlayerId': str})
    source = pd.read_csv(source_path, low_memory=False, dtype={'PlayerId': str})
    check_ids(linked, '학습 데이터', 'TargetSeason')
    check_ids(source, '원본 데이터', 'Season')
    if source['Season'].max() != 2025 or source['Season'].min() > 2015:
        raise ValueError('원본 데이터는 2015~2025년만 포함해야 합니다.')
    if (linked['TargetSeason'] > 2025).any():
        raise ValueError('학습 파일에 2026년 이후의 정답이 있습니다.')

    # Exactly the original 18 lag1 features; no target-year columns allowed.
    features = [c for c in linked.columns if c.endswith('_lag1')
                and c != 'InputQualified_lag1'
                and pd.to_numeric(linked[c], errors='coerce').notna().any()]
    if len(features) != 18:
        raise ValueError(f'{kind}: 기존 v2와 입력변수 수가 다릅니다 ({len(features)}개): {features}')
    for c in features:
        if not c.endswith('_lag1'):
            raise ValueError(f'미래 정보로 의심되는 입력변수: {c}')

    # Check original v2 features and alpha when its info JSON is available.
    info_path = ROOT / 'ml_results_v2' / f'{kind}_experiment_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8-sig'))
        if features != info['features']:
            raise ValueError(f'{kind}: v2 학습 입력변수 목록/순서가 다릅니다.')
        if not np.isclose(cfg['alpha'], info['chosen_hyperparameters']['ridge']['alpha']):
            raise ValueError(f'{kind}: v2 Ridge alpha가 다릅니다.')

    train = linked[linked['TargetSeason'].between(2016, 2025)].copy()
    train[cfg['target']] = pd.to_numeric(train[cfg['target']], errors='coerce')
    train = train[train[cfg['target']].notna()].copy()
    if train.empty:
        raise ValueError(f'{kind}: 학습 표본이 없습니다.')

    # 2025 qualification only; never use 2026 playing time to choose players.
    current = source[source['Season'] == 2025].copy()
    if kind == 'pitcher':
        current['IP_outs'] = ip_to_outs(current['IP'])
        current['GS_rate'] = (pd.to_numeric(current['GS'], errors='coerce') / pd.to_numeric(current['G'], errors='coerce').replace(0, np.nan))
    volume = pd.to_numeric(current[cfg['volume']], errors='coerce')
    current = current[volume >= cfg['min_volume']].copy()
    if current.empty:
        raise ValueError(f'{kind}: 2025 출전 기준 충족 선수가 없습니다.')

    # Construct 2026 lag1 inputs directly from 2025 records.
    Xnew = pd.DataFrame(index=current.index)
    for feature in features:
        original = feature[:-5]  # strip '_lag1'
        if original not in current.columns:
            raise ValueError(f'{kind}: 원본 CSV에 {original} 열이 없습니다.')
        Xnew[feature] = pd.to_numeric(current[original], errors='coerce')
    Xtrain = train[features].apply(pd.to_numeric, errors='coerce')
    if Xnew.isna().all(axis=0).any():
        raise ValueError(f'{kind}: 2025 입력 열 전체가 결측입니다: {Xnew.columns[Xnew.isna().all()].tolist()}')

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=cfg['alpha'])),
    ])
    model.fit(Xtrain, train[cfg['target']])
    predicted = model.predict(Xnew)
    out = pd.DataFrame({
        'PlayerId': current['PlayerId'].to_numpy(),
        'Name': current['Name'].to_numpy(),
        'Team_2025': current['Team'].to_numpy(),
        'InputSeason': 2025,
        'PredictionSeason': 2026,
        f'actual_2025_{cfg["metric"]}': pd.to_numeric(current[cfg['metric']], errors='coerce').to_numpy(),
        f'predicted_2026_{cfg["metric"]}': predicted,
        '2025_PA' if kind == 'batter' else '2025_IP_outs':
            pd.to_numeric(current[cfg['volume']], errors='coerce').to_numpy(),
    })
    if out.duplicated('PlayerId').any():
        raise ValueError(f'{kind}: 2026 예측 선수 중복')
    filename = OUT / f'{kind}_2026_predictions.csv'
    out.to_csv(filename, index=False, encoding='utf-8-sig', float_format='%.6f')
    print(f'[{kind}] 학습 {len(train)}개 선수-시즌 / 2026 예측 {len(out)}명 -> {filename}')
    return {'type': kind, 'train_rows': len(train), 'predicted_players': len(out),
            'features': features, 'ridge_alpha': cfg['alpha'],
            'train_source': str(linked_path.relative_to(ROOT)),
            'prediction_source': str(source_path.relative_to(ROOT)),
            'selection': '2025 season playing time only',
            '2026_actuals_used': False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    details = [run(kind, cfg) for kind, cfg in CONFIG.items()]
    pd.DataFrame([{k: v for k, v in row.items() if k != 'features'} for row in details]).to_csv(
        OUT / 'prediction_summary.csv', index=False, encoding='utf-8-sig')
    (OUT / 'prediction_manifest.json').write_text(json.dumps({
        'purpose': '2026 next-season forecast; 2026 actual statistics never loaded',
        'model': 'Ridge, fixed v2 2024 validation settings',
        'training_labels_through': 2025,
        'forecast_inputs_from': 2025,
        'population': 'players qualified in 2025; 2026 qualification unknown',
        'limitation': 'training labels come from historical target-year-qualified sample; distribution may differ from all 2025-qualified players',
        'details': details,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('완료. 2026 실제 성적과의 비교는 별도 평가 코드에서 수행하세요.')


if __name__ == '__main__':
    main()

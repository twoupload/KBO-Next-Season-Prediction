"""Evaluate precomputed 2026 KBO forecasts against held-out 2026 actuals.
Run from project root: python evaluate_2026.py
Requires pandas, numpy, matplotlib, scikit-learn.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'evaluation_2026'
OUT.mkdir(exist_ok=True)

def locate(filename, subdirs=('', 'results/predictions_2026', 'predictions_2026', 'data')):
    for folder in subdirs:
        path = ROOT / folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f'{filename} not found. Searched: ' + ', '.join(str(ROOT / x / filename) for x in subdirs))

def innings_to_outs(value):
    if pd.isna(value): return np.nan
    s = str(value).strip()
    if not s: return np.nan
    if '.' not in s: return int(s) * 3
    whole, fraction = s.split('.', 1)
    if fraction not in ('0', '1', '2', '00'):
        raise ValueError(f'Unexpected baseball IP value: {value!r}')
    return int(whole) * 3 + int(fraction)

def process(kind, target, threshold):
    pred = pd.read_csv(locate(f'{kind}_2026_predictions.csv'), dtype={'PlayerId': str})
    actual = pd.read_csv(locate(f'2026_total_{kind}.csv'), dtype={'PlayerId': str})
    pred_col = f'predicted_2026_{target}'
    prev_col = f'actual_2025_{target}'
    needed_pred = ['PlayerId','Name',prev_col,pred_col]
    needed_actual = ['PlayerId',target, 'PA' if kind=='batter' else 'IP']
    for col in needed_pred:
        if col not in pred: raise KeyError(f'Missing prediction column: {col}')
    for col in needed_actual:
        if col not in actual: raise KeyError(f'Missing actual column: {col}')
    if pred.PlayerId.isna().any() or pred.PlayerId.duplicated().any():
        raise ValueError(f'Invalid/duplicate PlayerId in {kind} predictions')
    actual = actual.copy()
    actual['playing_time_2026'] = (pd.to_numeric(actual['PA'], errors='coerce') if kind=='batter'
                                    else actual['IP'].map(innings_to_outs))
    cutoff = threshold if kind=='batter' else threshold * 3
    # A player may have multiple team rows; select TOTAL row if provided.
    if actual.PlayerId.duplicated().any():
        total = actual['Team'].astype(str).str.contains('total|합계|전체', case=False, regex=True)
        actual['_total'] = total
        actual = actual.sort_values(['_total','playing_time_2026'], ascending=[False,False])
        duplicates = actual[actual.PlayerId.duplicated(keep=False)]
        if not duplicates.empty:
            print(f'WARNING: {kind}: duplicate actual PlayerId; using TOTAL row if present, otherwise largest playing-time row. Review these IDs: {duplicates.PlayerId.unique().tolist()[:15]}')
        actual = actual.drop_duplicates('PlayerId', keep='first')
    actual = actual[['PlayerId',target,'playing_time_2026']].rename(columns={target:f'actual_2026_{target}'})
    merged = pred.merge(actual, on='PlayerId', how='left', validate='one_to_one', indicator=True)
    merged['2026_status'] = np.select([
        merged['_merge'].eq('left_only'),
        merged['playing_time_2026'].isna(),
        merged['playing_time_2026'].lt(cutoff),
        pd.to_numeric(merged[f'actual_2026_{target}'], errors='coerce').isna(),
    ], ['not_in_2026_file','playing_time_missing','below_2026_threshold','actual_metric_missing'], default='qualified_2026')
    merged = merged.drop(columns=['_merge'])
    merged[pred_col] = pd.to_numeric(merged[pred_col],errors='coerce')
    merged[prev_col] = pd.to_numeric(merged[prev_col],errors='coerce')
    merged[f'actual_2026_{target}'] = pd.to_numeric(merged[f'actual_2026_{target}'],errors='coerce')
    if merged[pred_col].isna().any(): raise ValueError('Missing predicted values')
    merged['error_pred_minus_actual'] = merged[pred_col] - merged[f'actual_2026_{target}']
    merged['absolute_error'] = merged['error_pred_minus_actual'].abs()
    merged['baseline_absolute_error'] = (merged[prev_col] - merged[f'actual_2026_{target}']).abs()
    merged.to_csv(OUT / f'{kind}_2026_all_forecasts_with_actuals.csv',index=False,encoding='utf-8-sig')
    evaluated = merged[merged['2026_status'].eq('qualified_2026')].copy()
    evaluated = evaluated.sort_values('absolute_error')
    evaluated.to_csv(OUT / f'{kind}_2026_qualified_comparison.csv',index=False,encoding='utf-8-sig')
    status_counts = merged['2026_status'].value_counts().to_dict()
    n = len(evaluated)
    if n == 0: raise ValueError(f'No qualified players for {kind}')
    y = evaluated[f'actual_2026_{target}'].to_numpy()
    yhat = evaluated[pred_col].to_numpy()
    baseline = evaluated[prev_col].to_numpy()
    metrics = dict(type=kind, target=target, forecast_count=len(pred), evaluated_count=n,
        excluded_count=len(pred)-n, mae=mean_absolute_error(y,yhat),
        rmse=np.sqrt(mean_squared_error(y,yhat)), r2=r2_score(y,yhat) if n>1 else np.nan,
        baseline_mae=mean_absolute_error(y,baseline), baseline_rmse=np.sqrt(mean_squared_error(y,baseline)),
        baseline_r2=r2_score(y,baseline) if n>1 else np.nan,
        mae_improvement_vs_baseline_pct=(1-mean_absolute_error(y,yhat)/mean_absolute_error(y,baseline))*100 if mean_absolute_error(y,baseline) else np.nan,
        mean_signed_error=float(np.mean(yhat-y)), **{f'status_{k}':v for k,v in status_counts.items()})
    # Scatter plot: actual vs forecast, with identity reference line
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(y,yhat,alpha=.65,s=28)
    bounds=[min(y.min(),yhat.min()),max(y.max(),yhat.max())]
    ax.plot(bounds,bounds,'--',color='gray',linewidth=1,label='Perfect prediction')
    ax.set(xlabel=f'2026 actual {target}',ylabel=f'2026 predicted {target}',title=f'{kind.title()} 2026: actual vs predicted (n={n})')
    ax.legend();fig.tight_layout();fig.savefig(OUT/f'{kind}_2026_scatter.png',dpi=180);plt.close(fig)
    # Show 12 largest errors, to avoid cherry-picking successful cases
    worst=evaluated.nlargest(min(12,n),'absolute_error').iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,max(4,len(worst)*.43)))
    labels=[str(x) for x in worst['Name']]
    ypos=np.arange(len(worst))
    ax.barh(ypos,worst['absolute_error'].to_numpy())
    ax.set_yticks(ypos,labels)
    ax.set_xlabel(f'Absolute error in {target}')
    ax.set_title(f'{kind.title()} 2026: largest prediction errors')
    fig.tight_layout();fig.savefig(OUT/f'{kind}_2026_largest_errors.png',dpi=180);plt.close(fig)
    print(f'{kind}: forecast={len(pred)}, qualified={n}, excluded={len(pred)-n}, MAE={metrics["mae"]:.4f}, baseline MAE={metrics["baseline_mae"]:.4f}')
    return metrics

if __name__=='__main__':
    rows=[process('batter','wRC+',100),process('pitcher','FIP',30)]
    pd.DataFrame(rows).to_csv(OUT/'2026_metrics_summary.csv',index=False,encoding='utf-8-sig')
    print('Saved outputs in',OUT)

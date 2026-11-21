#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KBO v2/v3/v4 experiment reporting. No model fitting or test-based tuning.

Usage: python analyze_model_results.py
       python analyze_model_results.py --project-dir C:\\Users\\LEE\\Desktop\\KBO_Prediction
       python analyze_model_results.py --v2-dir ... --v3-dir ... --v4-dir ...
Dependencies: pip install pandas numpy matplotlib
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['baseline', 'ridge', 'random_forest', 'xgboost']
LABELS = {'baseline':'Previous season', 'ridge':'Ridge', 'random_forest':'Random Forest', 'xgboost':'XGBoost'}
TARGETS = {'batter':'wRC+', 'pitcher':'FIP'}


def read_csv(path, required=()):
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    return df


def locate(directory, name, aliases=()):
    for filename in (name, *aliases):
        path = directory / filename
        if path.is_file():
            return path
    return None


def save_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'[saved] {path}')


def chart_setup():
    plt.rcParams.update({'figure.dpi': 125, 'savefig.dpi': 170, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})


def plot_models(v2, out):
    if v2 is None: return
    for target in ('batter', 'pitcher'):
        subset = v2[(v2.type == target) & (v2.split == 'test')].set_index('model')
        subset = subset.reindex([m for m in MODELS if m in subset.index])
        if subset.empty: continue
        fig, ax = plt.subplots(figsize=(8, 4.7))
        bars = ax.bar([LABELS.get(x,x) for x in subset.index], subset.MAE)
        for b, value in zip(bars, subset.MAE):
            ax.annotate(f'{value:.3f}', (b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylim(0, float(subset.MAE.max()) * 1.2)
        ax.set_ylabel(f'Test MAE ({TARGETS[target]}) — lower is better')
        ax.set_title(f'2025 Test: model comparison ({target})')
        fig.tight_layout(); fig.savefig(out / f'v2_{target}_model_comparison.png'); plt.close(fig)


def plot_history(v3, out):
    if v3 is None: return
    for target in ('batter','pitcher'):
        d = v3[(v3.type==target) & (v3.cohort=='common') & (v3.split=='test') &
               (v3.model.isin(['ridge','random_forest','xgboost']))]
        if d.empty: continue
        fig, ax = plt.subplots(figsize=(8,4.7))
        for model, g in d.groupby('model'):
            g = g.sort_values('lag')
            ax.plot(g.lag, g.MAE, marker='o', label=LABELS.get(model,model))
        ax.set_xticks([1,2,3]); ax.set_xlabel('Years of past data (lag)')
        ax.set_ylabel(f'2025 Test MAE ({TARGETS[target]}) — lower is better')
        ax.set_title(f'History length, SAME players ({target})')
        ax.legend(); ax.grid(axis='y', alpha=.2)
        fig.tight_layout(); fig.savefig(out / f'v3_{target}_common_history.png'); plt.close(fig)


def plot_ablation(v4, out):
    if v4 is None: return
    for split in ('validation','test'):
        d=v4[(v4.split==split) & (v4.model.isin(['ridge','random_forest','xgboost']))]
        if d.empty: continue
        fig,ax=plt.subplots(figsize=(9,4.7))
        variants=['all','no_gs_rate','no_gs_and_rate']
        x=np.arange(3); width=.25
        for i,model in enumerate(['ridge','random_forest','xgboost']):
            vals=[]
            for v in variants:
                sub=d[(d.model==model)&(d.variant==v)]
                vals.append(float(sub.MAE.iloc[0]) if len(sub) else np.nan)
            ax.bar(x+(i-1)*width,vals,width,label=LABELS[model])
        ax.set_xticks(x,['All features','Exclude GS_rate','Exclude GS + GS_rate'])
        ax.set_ylabel('FIP MAE — lower is better'); ax.set_title(f'Role feature ablation: {split} MAE')
        ax.legend(); ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out / f'v4_role_ablation_{split}.png');plt.close(fig)


def plot_scatter_and_errors(v3dir, out, summary):
    for target in ('batter','pitcher'):
        path=v3dir/f'{target}_native_lag1_test_predictions.csv'
        if not path.exists():
            warnings.warn(f'No prediction file: {path}'); continue
        df=read_csv(path,['PlayerId','TargetSeason','actual'])
        if df[['PlayerId','TargetSeason']].duplicated().any():
            raise ValueError(f'Duplicate player-season in {path}')
        for model in ['ridge','random_forest','xgboost']:
            pred=f'{model}_prediction'
            if pred not in df.columns: continue
            clean=df.dropna(subset=['actual',pred]).copy()
            if clean.empty: continue
            clean['absolute_error']=(clean.actual-clean[pred]).abs()
            clean['signed_error']=clean[pred]-clean.actual
            top=clean.nlargest(15,'absolute_error')
            cols=[x for x in ['PlayerId','TargetSeason','TargetName','TargetTeam','actual',pred,'signed_error','absolute_error'] if x in top.columns]
            save_csv(top[cols],out/f'largest_errors_{target}_{model}.csv')
            fig,ax=plt.subplots(figsize=(6,5.6))
            ax.scatter(clean.actual,clean[pred],alpha=.6,s=26)
            lo=min(clean.actual.min(),clean[pred].min());hi=max(clean.actual.max(),clean[pred].max())
            ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1.1,label='Perfect prediction')
            ax.set_xlabel(f'Actual {TARGETS[target]} (2025)')
            ax.set_ylabel(f'Predicted {TARGETS[target]}')
            ax.set_title(f'{target} — {LABELS[model]} | MAE={clean.absolute_error.mean():.3f}')
            ax.legend();fig.tight_layout();fig.savefig(out/f'scatter_{target}_{model}.png');plt.close(fig)
            summary.append({'type':target,'model':model,'n':len(clean),
                            'MAE_recomputed':clean.absolute_error.mean(),
                            'bias_pred_minus_actual':clean.signed_error.mean(),
                            'median_absolute_error':clean.absolute_error.median()})


def common_cohort_checks(v3dir, out):
    checks=[]
    for target in ('batter','pitcher'):
        for split in ('validation','test'):
            datasets=[]
            for lag in (1,2,3):
                path=v3dir/f'{target}_common_lag{lag}_{split}_predictions.csv'
                if not path.exists(): break
                d=read_csv(path,['PlayerId','TargetSeason','actual'])
                if d[['PlayerId','TargetSeason']].duplicated().any():
                    raise ValueError(f'Duplicate PlayerId/TargetSeason in {path.name}')
                datasets.append(d.set_index(['PlayerId','TargetSeason']).sort_index())
            if len(datasets)!=3: continue
            same_ids=all(datasets[0].index.equals(d.index) for d in datasets[1:])
            same_y=all(np.allclose(datasets[0].actual,d.actual,rtol=0,atol=1e-8) for d in datasets[1:]) if same_ids else False
            checks.append({'type':target,'split':split,'n':len(datasets[0]),
                           'same_player_season':same_ids,'same_actual':same_y})
            if not (same_ids and same_y):
                warnings.warn(f'Common cohort mismatch: {target} {split}')
    if checks:save_csv(pd.DataFrame(checks),out/'common_cohort_integrity.csv')


def compare_v2_v3(v2,v3,out):
    if v2 is None or v3 is None:return
    d=v3[(v3.cohort=='native') & (v3.lag==1)]
    keys=['type','split','model']
    merged=v2.merge(d,on=keys,suffixes=('_v2','_v3'))
    for m in ['MAE','RMSE','R2']:
        merged[f'{m}_difference_v3_minus_v2']=merged[f'{m}_v3']-merged[f'{m}_v2']
    save_csv(merged,out/'v2_v3_lag1_consistency.csv')


def model_selected_on_validation(df, grouping):
    val=df[df.split=='validation'].copy()
    if val.empty:return pd.DataFrame()
    val=val[val.model!='baseline']
    val=val.sort_values(['MAE','model'])
    best=val.groupby(grouping,as_index=False).first()
    return best


def main():
    parser=argparse.ArgumentParser(description='Create thesis-ready reports from existing v2-v4 results (no training).')
    parser.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--v2-dir',type=Path)
    parser.add_argument('--v3-dir',type=Path)
    parser.add_argument('--v4-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.project_dir.resolve()
    v2dir=args.v2_dir or root/'ml_results_v2'
    v3dir=args.v3_dir or root/'ml_results_v3'
    v4dir=args.v4_dir or root/'ml_results_v4'
    out=args.output_dir or root/'analysis_results'
    out.mkdir(parents=True,exist_ok=True)
    chart_setup()
    sources={}
    for label,directory in [('v2',v2dir),('v3',v3dir),('v4',v4dir)]:
        p=locate(directory,'metrics_summary.csv')
        if p is None:
            warnings.warn(f'{label} summary missing: {directory / "metrics_summary.csv"}; skipping that part')
            sources[label]=None
        else:
            sources[label]=read_csv(p,['type','split','model','n','MAE','RMSE','R2'])
            print(f'[loaded] {label}: {p} ({len(sources[label])} rows)')
    v2,v3,v4=sources['v2'],sources['v3'],sources['v4']
    if all(x is None for x in (v2,v3,v4)):
        sys.exit('No result summaries found. Place this script at the project root, or specify --project-dir.')
    if v2 is not None:
        save_csv(v2.sort_values(['type','split','MAE']),out/'table_v2_model_metrics.csv')
        selected=model_selected_on_validation(v2,['type'])
        save_csv(selected,out/'v2_best_model_by_2024_validation.csv')
        plot_models(v2,out)
    if v3 is not None:
        save_csv(v3.sort_values(['type','cohort','lag','split','MAE']),out/'table_v3_history_metrics.csv')
        selected=model_selected_on_validation(v3,['type','cohort'])
        save_csv(selected,out/'v3_best_setting_by_2024_validation.csv')
        common_cohort_checks(v3dir,out)
        plot_history(v3,out)
    if v4 is not None:
        save_csv(v4.sort_values(['split','model','variant']),out/'table_v4_role_ablation_metrics.csv')
        selected=model_selected_on_validation(v4,['type','cohort','lag'])
        save_csv(selected,out/'v4_best_setting_by_2024_validation.csv')
        plot_ablation(v4,out)
        pivot=v4.pivot_table(index=['split','model'],columns='variant',values='MAE').reset_index()
        if 'all' in pivot.columns:
            for variant in ['no_gs_rate','no_gs_and_rate']:
                if variant in pivot.columns:
                    pivot[f'{variant}_minus_all']=pivot[variant]-pivot['all']
        save_csv(pivot,out/'v4_ablation_mae_differences.csv')
    compare_v2_v3(v2,v3,out)
    errors=[]
    if v3 is not None:plot_scatter_and_errors(v3dir,out,errors)
    if errors:save_csv(pd.DataFrame(errors),out/'prediction_error_diagnostics.csv')
    report=[
        'KBO PLAYER NEXT-SEASON PERFORMANCE: EXPERIMENT REPORT',
        '=====================================================',
        'Scope: qualified players only (hitters >=100 PA; pitchers >=30 IP).',
        'IMPORTANT: qualification includes TARGET season; results are conditional on future playing time.',
        'Rows represent player-season observations, not necessarily distinct players.',
        '2024 validation: tuning/selection; 2025 test: held-out evaluation only.',
        'Do not select a new model by looking for the lowest 2025 test error.',
        'Batter wRC+ and pitcher FIP MAEs have DIFFERENT units; do not compare directly.',
        'v3 native lag sizes differ; only v3 common cohort isolates history-length effects.',
        'Small common-cohort test sizes mean apparent improvements may be noisy.',
        'Ablation reflects predictive contribution, not causal effects of pitcher role.',
        '',
    ]
    for label,df,grouping in [('V2 model comparison',v2,['type']),
                              ('V3 common history comparison',v3,['type','cohort']),
                              ('V4 role ablation',v4,['type','cohort','lag'])]:
        report.append(label.upper());report.append('-'*len(label))
        if df is None:report.append('Data not found.\n');continue
        d=df.copy()
        if label.startswith('V3'):d=d[d.cohort=='common']
        best=model_selected_on_validation(d,grouping)
        if best.empty:report.append('No validation records.\n');continue
        for _,r in best.iterrows():
            parts=[f'{key}={r[key]}' for key in grouping]
            parts.extend([f'model={r.model}',f'validation MAE={r.MAE:.4f}'])
            if 'lag' in r.index and 'lag' not in grouping:parts.append(f'lag={int(r.lag)}')
            if 'variant' in r.index:parts.append(f'variant={r.variant}')
            report.append(' | '.join(parts))
        report.append('')
    (out/'README_results.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    print(f'\nDONE. Output directory: {out.resolve()}')

if __name__=='__main__':main()

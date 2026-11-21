"""Leakage-safe KBO league-mean and Marcel-inspired baselines.

This is a documented adaptation, NOT the original MLB Marcel algorithm.
Only seasons < target year are used for each forecast. No tuning on test outcomes.
Run: python compare_baselines.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'baseline_comparison'
CONFIG={'batter':('wRC+','PA',100,1200.0), 'pitcher':('FIP','IP_outs',90,1080.0)}
RECENCY={1:5.,2:4.,3:3.}

def ip_outs(v):
    if pd.isna(v):return np.nan
    s=str(v).strip().split('.')
    frac=int(s[1]) if len(s)>1 else 0
    if frac not in (0,1,2):raise ValueError(f'Invalid IP: {v}')
    return int(s[0])*3+frac

def load(kind):
    p=ROOT/'data'/f'kbo_{kind}_2015_2025_active.csv'
    d=pd.read_csv(p,dtype={'PlayerId':str},low_memory=False)
    if kind=='pitcher':d['IP_outs']=d['IP'].map(ip_outs)
    metric,volume,_,_=CONFIG[kind]
    for c in ['Age',metric,volume]:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['Season']=pd.to_numeric(d['Season']).astype(int)
    assert not d.duplicated(['PlayerId','Season']).any()
    return d

def league_mean(d,kind,year):
    metric,volume,_,_=CONFIG[kind]
    x=d[(d.Season==year)&d[metric].notna()&(d[volume]>0)]
    if x.empty:raise ValueError(f'No league stats for {year}')
    return float(np.average(x[metric],weights=x[volume]))

def project(d,kind,ids,target_year):
    metric,volume,threshold,prior=CONFIG[kind]
    assert 2016<=target_year<=2026
    # Use most recent available league season; never target-year actuals.
    league={lag:league_mean(d,kind,target_year-lag) for lag in RECENCY if target_year-lag>=2015}
    mean=sum(RECENCY[lag]*v for lag,v in league.items())/sum(RECENCY[lag] for lag in league)
    history=d[(d.Season<target_year)&(d.Season>=target_year-3)].copy()
    history=history[history[volume].gt(0)&history[metric].notna()]
    history['lag']=target_year-history.Season
    history['weighted_volume']=history[volume]*history.lag.map(RECENCY)
    history['weighted_value']=history.weighted_volume*history[metric]
    sums=history.groupby('PlayerId')[['weighted_volume','weighted_value']].sum()
    # Age at target season from most recent available record (no target-year info).
    age=d[(d.Season<target_year)&(d.Season>=target_year-3)].sort_values('Season').drop_duplicates('PlayerId',keep='last').set_index('PlayerId')
    result=[]
    for pid in ids:
        v=sums.loc[pid,'weighted_volume'] if pid in sums.index else 0.
        w=sums.loc[pid,'weighted_value'] if pid in sums.index else 0.
        raw=(w+prior*mean)/(v+prior)
        a=float(age.loc[pid,'Age']+(target_year-age.loc[pid,'Season'])) if pid in age.index and pd.notna(age.loc[pid,'Age']) else np.nan
        # Explicit KBO metric adaptation: adjust *deviation* from league mean,
        # not raw FIP/wRC+ levels; lower FIP is better.
        factor=1.0+(0.006*(29-a) if a<=29 else 0.003*(29-a)) if np.isfinite(a) else 1.
        if kind=='batter': adjusted=mean+(raw-mean)*factor
        else: adjusted=mean+(raw-mean)*factor
        result.append({'PlayerId':pid,'league_mean_prediction':mean,'marcel_adapted_prediction':adjusted,'marcel_pre_age':raw,'projected_age':a,'historical_weighted_volume':v,'marcel_reliability':v/(v+prior)})
    return pd.DataFrame(result)

def score(frame,kind,year):
    actual='actual'
    out=[]
    for label,col in [('League mean','league_mean_prediction'),('Marcel-inspired','marcel_adapted_prediction'),('Persistence','baseline_prediction'),('Ridge','ridge_prediction'),('Random Forest','random_forest_prediction'),('XGBoost','xgboost_prediction')]:
        if col not in frame:continue
        y=frame[actual].to_numpy(float); pred=frame[col].to_numpy(float)
        out.append({'kind':kind,'test_year':year,'model':label,'n':len(y),'MAE':mean_absolute_error(y,pred),'RMSE':np.sqrt(mean_squared_error(y,pred)),'R2':r2_score(y,pred)})
    return out

def main():
    OUT.mkdir(exist_ok=True)
    metrics=[]
    for kind in CONFIG:
        d=load(kind)
        for year,split in [(2024,'validation'),(2025,'test')]:
            f=pd.read_csv(ROOT/'ml_results_v2'/f'{kind}_{split}_predictions.csv',dtype={'PlayerId':str})
            p=project(d,kind,f.PlayerId.tolist(),year)
            m=f.merge(p,on='PlayerId',validate='one_to_one')
            assert len(m)==len(f)
            m.to_csv(OUT/f'{kind}_{year}_comparison.csv',index=False,encoding='utf-8-sig')
            metrics.extend(score(m,kind,year))
        f=pd.read_csv(ROOT/'evaluation_2026'/f'{kind}_2026_qualified_comparison.csv',dtype={'PlayerId':str})
        p=project(d,kind,f.PlayerId.tolist(),2026)
        m=f.merge(p,on='PlayerId',validate='one_to_one')
        assert len(m)==len(f)
        metric=CONFIG[kind][0]
        m['actual']=m[f'actual_2026_{metric}']
        m['baseline_prediction']=m[f'actual_2025_{metric}']
        m['ridge_prediction']=m[f'predicted_2026_{metric}']
        m.to_csv(OUT/f'{kind}_2026_comparison.csv',index=False,encoding='utf-8-sig')
        metrics.extend(score(m,kind,2026))
    result=pd.DataFrame(metrics)
    result.to_csv(OUT/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    print(result.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Saved to',OUT)

if __name__=='__main__':main()

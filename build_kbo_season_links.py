"""KBO 2015-2025: connect player seasons into leakage-safe candidate ML tables.
Usage: python build_kbo_season_links.py --input-dir /path/to/csvs --output-dir /path/to/output
Requires pandas. Input: kbo_batter_2015_2025_active.csv and kbo_pitcher_2015_2025_active.csv.
"""
import argparse
from pathlib import Path
import pandas as pd

METRICS = {
 'batter': dict(target='wRC+', volume='PA', threshold=100, features=['Age','PA','G','AB','H','HR','BB','SO','BB%','K%','AVG','OBP','SLG','OPS','ISO','BABIP','wOBA','wRC+']),
 'pitcher': dict(target='FIP', volume='IP_outs', threshold=90, features=['Age','IP_outs','G','GS','GS_rate','ERA','FIP','SO','BB','HR','K%','BB%','K-BB%','K/9','BB/9','HR/9','WHIP','BABIP']),
}

def ip_to_outs(v):
    if pd.isna(v) or str(v).strip()=='': return pd.NA
    s=str(v).strip()
    parts=s.split('.')
    whole=int(parts[0]); fraction=int(parts[1]) if len(parts)>1 else 0
    if fraction not in (0,1,2): raise ValueError(f'Invalid baseball IP notation: {v}')
    return 3*whole+fraction

def prepare(path, kind):
    d=pd.read_csv(path,low_memory=False)
    if not {'PlayerId','Season',METRICS[kind]['target']}.issubset(d.columns): raise ValueError(f'Missing required columns in {path}')
    d['Season']=pd.to_numeric(d['Season'],errors='raise').astype(int)
    if d[['PlayerId','Season']].isna().any().any(): raise ValueError('Missing player id / season')
    if d.duplicated(['PlayerId','Season']).any(): raise ValueError('Duplicate player-season; resolve before linking')
    if kind=='pitcher':
        d['IP_outs']=d['IP'].map(ip_to_outs).astype('Int64')
        d['GS_rate']=pd.to_numeric(d['GS'],errors='coerce')/pd.to_numeric(d['G'],errors='coerce').replace(0,float('nan'))
        d['IP_decimal']=d['IP_outs'].astype(float)/3
    else: d['PA']=pd.to_numeric(d['PA'],errors='coerce')
    d['target_available']=pd.to_numeric(d[METRICS[kind]['target']],errors='coerce').notna()
    d['target_qualified']=d['target_available'] & (d[METRICS[kind]['volume']]>=METRICS[kind]['threshold'])
    return d.sort_values(['PlayerId','Season']).reset_index(drop=True)

def link(d,kind,lag):
    cfg=METRICS[kind]
    base=d[['PlayerId','Season','Name','Team',cfg['target'],cfg['volume'],'target_qualified']].copy()
    base=base.rename(columns={'Season':'TargetSeason','Name':'TargetName','Team':'TargetTeam',cfg['target']:'Target_'+cfg['target'],cfg['volume']:'TargetVolume','target_qualified':'TargetQualified'})
    out=base.copy()
    for i in range(1,lag+1):
        f=d[['PlayerId','Season']+[c for c in cfg['features'] if c in d.columns]].copy()
        f['Season']=f['Season']+i
        f=f.rename(columns={'Season':'TargetSeason',**{c:f'{c}_lag{i}' for c in f.columns if c not in ('PlayerId','Season')}})
        out=out.merge(f,on=['PlayerId','TargetSeason'],how='inner',validate='one_to_one')
    # Feature volume conditions are an experimental choice, not hard-coded into all linked samples.
    for i in range(1,lag+1):
        col=f"{cfg['volume']}_lag{i}"
        out[f'InputQualified_lag{i}']=out[col]>=cfg['threshold']
    out['AllInputsQualified']=out[[f'InputQualified_lag{i}' for i in range(1,lag+1)]].all(axis=1)
    out['QualifiedBoth']=out['AllInputsQualified'] & out['TargetQualified']
    out=out.sort_values(['TargetSeason','PlayerId']).reset_index(drop=True)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=Path('.'));p.add_argument('--output-dir',type=Path,default=Path('kbo_linked_2015_2025'));args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for kind in ('batter','pitcher'):
        path=args.input_dir/f'data/kbo_{kind}_2015_2025_active.csv'
        d=prepare(path,kind)
        d.to_csv(args.output_dir/f'{kind}_source_checked.csv',index=False,encoding='utf-8-sig')
        for lag in (1,2,3):
            linked=link(d,kind,lag)
            stem=f'{kind}_lag{lag}'
            linked.to_csv(args.output_dir/f'{stem}_all_linked.csv',index=False,encoding='utf-8-sig')
            linked.loc[linked.QualifiedBoth].to_csv(args.output_dir/f'{stem}_qualified.csv',index=False,encoding='utf-8-sig')
            for year,g in linked.groupby('TargetSeason'):
                summary.append(dict(kind=kind,lag=lag,target_season=int(year),all_linked=len(g),qualified=int(g.QualifiedBoth.sum()),input_qualified=int(g.AllInputsQualified.sum()),target_qualified=int(g.TargetQualified.sum())))
    pd.DataFrame(summary).to_csv(args.output_dir/'summary_by_target_year.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(summary).groupby(['kind','lag'])[['all_linked','qualified']].sum().to_string())
    print('Output:',args.output_dir.resolve())
if __name__=='__main__':main()

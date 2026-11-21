# -*- coding: utf-8 -*-
"""KBO 2026 논문용 표·그림 생성. 기존 예측/평가 결과를 수정하거나 재학습하지 않음.
실행: python make_thesis_figures.py
선택: python make_thesis_figures.py --input-dir evaluation_2026 --output-dir thesis_figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent

def font_setup():
    preferred = ['Malgun Gothic', '맑은 고딕', 'AppleGothic', 'NanumGothic', '나눔고딕', 'NanumSquare', 'Noto Sans CJK KR', 'UnDotum']
    installed = {x.name for x in font_manager.fontManager.ttflist}
    selected = next((x for x in preferred if x in installed), None)
    if selected:
        plt.rcParams['font.family'] = selected
    else:
        print('[주의] 한글 글꼴을 찾지 못했습니다. Windows라면 맑은 고딕을 설치하세요.')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    return selected

def find_file(directory, name):
    p = directory / name
    if not p.exists():
        raise FileNotFoundError(f'{p} 파일을 찾지 못했습니다. --input-dir 경로를 확인하세요.')
    return p

def short_name(value):
    import re
    s = str(value).strip()
    # 원본은 보통 'English Name 한국어명' 형식. 마지막 연속 한글을 사용.
    korean = re.findall(r'[가-힣]+', s)
    return ''.join(korean) if korean else s

def save_fig(fig, out, stem):
    fig.savefig(out / f'{stem}.png', dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def metrics_from_rows(df, target):
    a = df[f'actual_2026_{target}'].to_numpy(float)
    p = df[f'predicted_2026_{target}'].to_numpy(float)
    prev = df[f'actual_2025_{target}'].to_numpy(float)
    mse = np.mean((p-a)**2)
    return {'n':len(df), 'mae':np.mean(np.abs(p-a)), 'rmse':np.sqrt(mse),
            'r2':1-np.sum((p-a)**2)/np.sum((a-a.mean())**2),
            'baseline_mae':np.mean(np.abs(prev-a))}

def scatter(df, kind, target, out):
    a=df[f'actual_2026_{target}'].to_numpy(float)
    p=df[f'predicted_2026_{target}'].to_numpy(float)
    lo=min(a.min(),p.min()); hi=max(a.max(),p.max()); margin=(hi-lo)*0.06
    fig,ax=plt.subplots(figsize=(7.3,6.4))
    ax.scatter(a,p,s=40,alpha=.72,edgecolors='white',linewidths=.5)
    ax.plot([lo-margin,hi+margin],[lo-margin,hi+margin],ls='--',color='#666666',label='완전 일치 (예측 = 실제)')
    ax.set(xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin),
           xlabel=f'2026년 실제 {target}',ylabel=f'2026년 예측 {target}',
           title=f'2026년 {kind} 실제값과 예측값 (n={len(df)})')
    ax.legend(loc='best');ax.grid(alpha=.16)
    save_fig(fig,out,f'{kind}_2026_실제_예측_산점도')

def error_bars(df,kind,target,out,n=10):
    d=df.nlargest(n,'absolute_error').iloc[::-1]
    names=[short_name(x) for x in d['Name']]
    fig,ax=plt.subplots(figsize=(9,6.5))
    ax.barh(range(len(d)),d['absolute_error'],color='#4a78a6')
    ax.set_yticks(range(len(d)),labels=names)
    ax.set_xlabel(f'절대오차 ({target})');ax.set_title(f'2026년 {kind} 예측 오차 상위 {len(d)}명')
    ax.grid(axis='x',alpha=.16)
    save_fig(fig,out,f'{kind}_2026_오차_상위')

def comparison_bars(df,kind,target,out):
    # 작은 오차/중간 오차/큰 오차를 균형 있게 선발; 동일 선수를 중복 선정하지 않음
    d=df.sort_values(['absolute_error','PlayerId']).reset_index(drop=True)
    idx=[0,len(d)//4,len(d)//2,3*len(d)//4,len(d)-1]
    chosen=d.iloc[sorted(set(idx))].copy()
    chosen['선수']=chosen['Name'].map(short_name)
    x=np.arange(len(chosen)); width=.26
    fig,ax=plt.subplots(figsize=(10,5.7))
    ax.bar(x-width,chosen[f'actual_2025_{target}'],width,label='2025 실제',color='#8b9fb2')
    ax.bar(x,chosen[f'predicted_2026_{target}'],width,label='2026 예측',color='#4a78a6')
    ax.bar(x+width,chosen[f'actual_2026_{target}'],width,label='2026 실제',color='#e3a05d')
    ax.set_xticks(x,chosen['선수']);ax.set_ylabel(target)
    ax.set_title(f'2026년 {kind} 선수별 성적 비교 (오차 분위별 대표 사례)')
    ax.legend(ncol=3);ax.grid(axis='y',alpha=.15)
    save_fig(fig,out,f'{kind}_2026_선수별_비교')
    return chosen

def top_performer_tables_and_figures(all_df, kind, target, out):
    """2025년 기록으로만 상위 10명을 확정. 2026년 결측/미달 선수도 제외하지 않음."""
    past=f'actual_2025_{target}'
    forecast=f'predicted_2026_{target}'
    actual=f'actual_2026_{target}'
    if all_df['PlayerId'].duplicated().any():
        raise ValueError(f'{kind}: 전체 예측 파일에 PlayerId 중복이 있습니다')
    required=['PlayerId','Name',past,forecast,actual,'2026_status','absolute_error']
    missing=[c for c in required if c not in all_df]
    if missing: raise ValueError(f'{kind}: 상위 선수 분석에 필요한 열 누락: {missing}')
    d=all_df.copy()
    for c in [past,forecast,actual,'absolute_error']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=[past,forecast])
    # wRC+는 클수록 우수, FIP는 작을수록 우수.
    d=d.sort_values([past,'PlayerId'],ascending=[kind=='투수',True],kind='stable').head(10).copy()
    d.insert(0,'2025_순위',range(1,len(d)+1))
    d.insert(2,'선수명_한글',d['Name'].map(short_name))
    d['2026_평가가능']=d['2026_status'].eq('qualified_2026')
    # 기존 평가 CSV와 일관되도록 기준 미달 실제 성적은 평가 결과로 제시하지 않음.
    d['2026_평가용_실제']=d[actual].where(d['2026_평가가능'])
    d['2026_평가용_절대오차']=d['absolute_error'].where(d['2026_평가가능'])
    columns=['2025_순위','PlayerId','선수명_한글','Name',past,forecast,
             '2026_평가용_실제','2026_평가용_절대오차','2026_status','2026_평가가능']
    for n in (10,5):
        part=d.head(n).copy()
        part[columns].to_csv(out/f'{kind}_2025성적상위{n}명_2026예측비교.csv',index=False,encoding='utf-8-sig')
        x=np.arange(len(part));w=.25
        fig,ax=plt.subplots(figsize=(max(9,n*1.25),6))
        ax.bar(x-w,part[past],w,label=f'2025 실제 {target}',color='#8b9fb2')
        ax.bar(x,part[forecast],w,label=f'2026 예측 {target}',color='#4a78a6')
        actual_values=part['2026_평가용_실제'].to_numpy(dtype=float)
        valid=np.isfinite(actual_values)
        ax.bar(x[valid]+w,actual_values[valid],w,label=f'2026 실제 {target} (평가 가능)',color='#e3a05d')
        # 미평가 선수를 시각적으로 명확하게 구분하고 0으로 오해하지 않게 함.
        if (~valid).any():
            for xi in x[~valid]:
                ax.text(xi+w,0,'미평가',rotation=90,va='bottom',ha='center',fontsize=9,color='#9b3d3d')
        ax.set_xticks(x,[f"{r['2025_순위']}위\n{r['선수명_한글']}" for _,r in part.iterrows()])
        ax.set_ylabel(target)
        direction='높은' if kind=='타자' else '낮은'
        ax.set_title(f'2025년 {target} {direction} 상위 {n}명: 2026년 예측과 실제')
        ax.legend(ncol=3,fontsize=10,loc='best');ax.grid(axis='y',alpha=.15)
        fig.text(.5,.01,'선수 선정: 2025년 성적 기준 / 미평가: 2026년 출전 기준 미달 또는 기록 없음',
                 ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.035,1,1))
        save_fig(fig,out,f'{kind}_2025성적상위{n}명_2026예측비교')
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='evaluation_2026')
    ap.add_argument('--output-dir',default='thesis_figures')
    args=ap.parse_args()
    indir=Path(args.input_dir);out=Path(args.output_dir)
    if not indir.is_absolute():indir=BASE/indir
    if not out.is_absolute():out=BASE/out
    out.mkdir(parents=True,exist_ok=True)
    print('한글 글꼴:',font_setup() or '없음')
    summary=pd.read_csv(find_file(indir,'2026_metrics_summary.csv'),encoding='utf-8-sig')
    rows=[]; pop=[]
    for kind,target,english in [('타자','wRC+','batter'),('투수','FIP','pitcher')]:
        df=pd.read_csv(find_file(indir,f'{english}_2026_qualified_comparison.csv'),encoding='utf-8-sig',dtype={'PlayerId':str})
        all_df=pd.read_csv(find_file(indir,f'{english}_2026_all_forecasts_with_actuals.csv'),encoding='utf-8-sig',dtype={'PlayerId':str})
        required=['PlayerId','Name',f'actual_2025_{target}',f'predicted_2026_{target}',f'actual_2026_{target}','absolute_error']
        missing=[x for x in required if x not in df.columns]
        if missing:raise ValueError(f'{english}: 필요한 열 누락 {missing}')
        if df['PlayerId'].duplicated().any():raise ValueError(f'{english}: PlayerId 중복')
        if not np.allclose(df['absolute_error'],np.abs(df[f'predicted_2026_{target}']-df[f'actual_2026_{target}']),atol=1e-6):
            raise ValueError(f'{english}: 절대오차 열 불일치')
        sm=summary.loc[summary['type']==english]
        if len(sm)!=1:raise ValueError(f'{english}: 요약 행이 없거나 중복')
        sm=sm.iloc[0]
        m=metrics_from_rows(df,target)
        if int(sm['evaluated_count'])!=len(df) or int(sm['forecast_count'])!=len(all_df) or abs(float(sm['mae'])-m['mae'])>1e-5:
            raise ValueError(f'{english}: 요약 파일과 선수별 비교 파일이 일치하지 않습니다')
        rows.append({'대상':kind,'예측지표':target,'예측선수수':len(all_df),'평가선수수':len(df),
                     '평가제외수':len(all_df)-len(df),'Ridge_MAE':m['mae'],'Baseline_MAE':m['baseline_mae'],
                     'MAE_개선율(%)':100*(m['baseline_mae']-m['mae'])/m['baseline_mae'],
                     'RMSE':m['rmse'],'R2':m['r2']})
        pop.append({'대상':kind,'예측':len(all_df),'평가':len(df),
                    '2026기준미달':int(sm['status_below_2026_threshold']),
                    '2026파일없음':int(sm['status_not_in_2026_file'])})
        d=df.copy();d.insert(1,'선수명_한글',d['Name'].map(short_name))
        d=d.sort_values('absolute_error')
        d.to_csv(out/f'{kind}_2026_전체_선수별_비교.csv',index=False,encoding='utf-8-sig')
        scatter(df,kind,target,out)
        error_bars(df,kind,target,out)
        chosen=comparison_bars(df,kind,target,out)
        top_performer_tables_and_figures(all_df,kind,target,out)
        chosen.to_csv(out/f'{kind}_2026_대표선수_비교.csv',index=False,encoding='utf-8-sig')
        # 오차 상·중·하 구간을 균형 있게 선정한 논문용 표
        slices=[('오차 작음',d.head(3)),('오차 중간',d.iloc[max(0,len(d)//2-1):len(d)//2+2]),('오차 큼',d.tail(3))]
        samples=[]
        for label,sub in slices:
            for _,r in sub.iterrows():
                samples.append({'구분':label,'선수명':short_name(r['Name']),'PlayerId':r['PlayerId'],
                                f'2025_실제_{target}':r[f'actual_2025_{target}'],
                                f'2026_예측_{target}':r[f'predicted_2026_{target}'],
                                f'2026_실제_{target}':r[f'actual_2026_{target}'],
                                '절대오차':r['absolute_error']})
        pd.DataFrame(samples).to_csv(out/f'{kind}_2026_논문용_대표사례표.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out/'표_2026_예측성능.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pop).to_csv(out/'표_2026_예측대상_평가대상.csv',index=False,encoding='utf-8-sig')
    # 비교 그래프: 타자와 투수의 단위가 달라 절대 MAE를 직접 비교하지 않고 개선율만 표시
    fig,ax=plt.subplots(figsize=(7,4.6))
    names=[r['대상'] for r in rows];imps=[r['MAE_개선율(%)'] for r in rows]
    bars=ax.bar(names,imps,color=['#4a78a6','#e3a05d'],width=.45)
    for b,v in zip(bars,imps):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.3,f'{v:.1f}%',ha='center')
    ax.set_ylim(0,max(imps)*1.25);ax.set_ylabel('직전 시즌 성적 유지 기준 대비 MAE 감소율 (%)')
    ax.set_title('2026년 Ridge 모델의 기준 예측 대비 개선율')
    ax.grid(axis='y',alpha=.15)
    save_fig(fig,out,'2026_기준예측_대비_개선율')
    (out/'README_논문활용.txt').write_text(
        'KBO 2026 논문용 결과\n\n'
        '1. 이 스크립트는 기존 2026년 평가 결과를 시각화하며 재학습하지 않습니다.\n'
        '2. 예측 대상은 2025년 기준 충족 선수, 성능 평가는 2026년에도 기준을 충족한 선수로 제한합니다.\n'
        '3. 대표 사례는 오차 크기에 따라 기계적으로 선택했습니다. 전체 성능의 근거는 전체 평가 표본의 MAE/RMSE/R²입니다.\n'
        '4. 타자 wRC+와 투수 FIP의 절대 MAE는 단위가 다르므로 직접 비교하지 마세요.\n'
        '5. 2026년은 사전 등록된 실시간 예측이 아닌 시간 순서 기반 추가 시즌 검증입니다.\n'
        '6. PNG는 문서 삽입용, PDF는 벡터 그림용입니다. CSV는 UTF-8 BOM 인코딩입니다.\n'
        '7. 추가된 상위 5명/10명은 2025년 wRC+ 내림차순 또는 FIP 오름차순으로 예측 대상 전체에서 먼저 확정합니다.\n'
        '8. 2026년 평가 기준 미달/기록 없음은 상위 명단에 남기되 실제값을 미평가로 표시합니다. 2026년 성적을 기준으로 선수를 고르지 않습니다.\n',encoding='utf-8')
    print('완료:',out)
    for r in rows:print(f"{r['대상']}: 평가 {r['평가선수수']}명 / MAE {r['Ridge_MAE']:.4f} / 개선 {r['MAE_개선율(%)']:.2f}%")

if __name__=='__main__':main()

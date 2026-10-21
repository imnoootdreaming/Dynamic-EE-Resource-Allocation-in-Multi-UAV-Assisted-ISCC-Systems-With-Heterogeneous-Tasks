"""Independent arithmetic/matching checks and descriptive plots; no solver."""
from pathlib import Path
import csv, hashlib, itertools, json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
OUTER=OUT.parents[3]
LOG=OUTER/'logs/MHBPPO/2026-10-19_09-44-15'
OLD=OUT.parent/'mhbppo_matched_2env_20261020'

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    n=r['freeze']['last_update'];lo=n-499
    with (OUT/'episodes.csv').open(newline='',encoding='utf8') as f:
        episodes=list(csv.DictReader(f))
    checks={}
    for name,w in r['windows'].items():
        a,b=w['range'];e=[x for x in episodes if a<=int(x['update'])<=b]
        checks[name+'_episode_counts']=(len(e)==w['episodes'] and
            sum(int(x['optimal'])==40 for x in e)==w['all_optimal_n'] and
            sum(float(x['completion'])==100 for x in e)==w['all_completion_n'])
    d=pd.read_csv(OUT/'latest_steps.csv',keep_default_na=False)
    f=pd.read_csv(OUT/'failed_step_details.csv',keep_default_na=False)
    checks['optimal_slot_total']=int((d.status=='optimal').sum())==38223
    checks['all_optimal_by_raw_steps']=int(d.groupby(['update','worker']).failure.any().eq(False).sum())==236
    checks['all_completion_by_raw_steps']=int(d.groupby(['update','worker']).success.all().sum())==202
    checks['all_infeasible_sensing_certified']=bool(d.loc[d.status=='infeasible','sensing_infeasible'].all())
    checks['versions_all_steps']=bool((d.wv==d['update']-1).all())
    # Exhaustive injective assignments independently check the augmenting-path
    # matching routine and persisted threshold flags. Only 10P4=5040 per matrix.
    assignments=np.asarray(list(itertools.permutations(range(10),4)))
    uavs=np.arange(4)[None,:]
    detail=f.set_index(['update','worker','slot']);matching_checks=[];floor_sources=Counter();status_sources=Counter()
    snapshots=OUT/'frozen/failed_diagnostics.jsonl'
    with snapshots.open('wb') as out:
        for name,g in d[d.failure].groupby('file'):
            with (LOG/name).open('rb') as source:
                for row in g.sort_values('offset').itertuples():
                    source.seek(int(row.offset));raw=source.readline()
                    assert hashlib.sha256(raw).hexdigest()==row.raw_sha256
                    v=json.loads(raw);out.write(raw)
                    s=v['solver'];status_sources[(s['solver_status_kind'],s['raw_solver_status_kind'],s['status_source'])]+=1
                    c=v['sensing']['candidates'];z=detail.loc[(row.update,row.worker,row.slot)]
                    floor_sources[c['cu_power_floor_source']]+=1
                    for power,prefix in [('current_power','current_power'),('min_power','min_power')]:
                        m=np.asarray(c[power+'_margin'])
                        for threshold,key in [(1+1e-6,prefix+'_rematch_possible'),(2,prefix+'_margin2_matching')]:
                            exists=bool((m[uavs,assignments]>=threshold).all(axis=1).any())
                            matching_checks.append(exists==bool(z[key]))
    checks['all_failed_raw_sha_verified']=True
    checks['exhaustive_matching_checks']=all(matching_checks)
    checks['legal_floor']=dict(floor_sources)=={'environment_action_space':1777}
    sources=[]
    for item in json.loads((OLD/'source_check.json').read_text(encoding='utf8')):
        sha=hashlib.sha256((OUTER/item['path']).read_bytes()).hexdigest()
        sources.append({'path':item['path'],'sha256':sha,'match':sha==item['expected']})
    checks['production_sources_unchanged']=all(x['match'] for x in sources)
    assert all(checks.values()),checks
    infeasible=f[f.status=='infeasible']
    findings={'infeasible_only_min_margin_median':float(infeasible.margin_min.median()),
              'infeasible_only_single_uav_n':int((infeasible.failed_uav_count==1).sum()),
              'infeasible_only_current_power_rematch_n':int(infeasible.current_power_rematch_possible.sum()),
              'failed_steps_mean_Q':float(d.loc[d.failure,'Q'].mean()),
              'one_failed_uav_M068_Q':float((-np.log(.68))/(-np.log(.68)+np.log(10))/4),
              'optimal_one_failure_repair_needed_pct':(500-236)/287*100,
              'whole_completion_with_all_single_failures_repaired_pct':(202+274)/10,
              'status_sources':[{'resolved':k[0],'raw':k[1],'source':k[2],'n':v} for k,v in status_sources.items()]}
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'frozen').iterdir()}
    (OUT/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (OUT/'validation.json').write_text(json.dumps({'passed':True,'checks':checks,'matching_checks':len(matching_checks),
       'additional_findings':findings,'production_sources':sources,'solver_calls':0,'optimizer_steps':0},indent=2),encoding='utf8')

    e=pd.read_csv(OUT/'episodes.csv');u=e.groupby('update')[['all_optimal','all_completion','optimal_failures']].mean()
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for col,label,color in [('all_optimal','40槽全部 optimal','#087F8C'),('all_completion','40槽全部 legacy completion','#D17C00')]:
        ax.plot(u.index,u[col].rolling(500,min_periods=500).mean()*100,label=label,color=color,lw=2)
    ax.axhline(50,color='#B64949',ls='--',lw=1.3,label='50% 目标')
    ax.scatter([n],[35],color='#087F8C',s=40,zorder=5)
    ax.annotate('末100更新：35%\n短窗口，不能替代500更新均值',xy=(n,35),xytext=(3300,37),fontsize=9,
                arrowprops={'arrowstyle':'->','color':'#666'})
    ax.set(title='整回合成功率：500更新滚动窗口（1000回合）',xlabel='update',ylabel='回合比例 (%)',ylim=(0,55))
    ax.legend(loc='upper left',fontsize=8)
    ax=axes[0,1]
    ax.plot(u.index,u.optimal_failures.rolling(500,min_periods=500).mean(),color='#5B5EA6',lw=2)
    ax.set(title='每回合非 optimal 槽数：500更新滚动均值',xlabel='update',ylabel='非 optimal 槽数',ylim=(0,None))
    ax=axes[1,0]
    last=e[e['update'].between(lo,n)];h=last.optimal_failures.value_counts()
    heights=[h.get(i,0) for i in range(6)]+[int((last.optimal_failures>=6).sum())]
    bars=ax.bar(['0','1','2','3','4','5','≥6'],heights,color=['#087F8C','#D17C00']+['#A0ABB8']*5)
    ax.bar_label(bars,padding=3)
    ax.set(title=f'末1000回合：236条零失败，287条仅失败1槽',xlabel='每回合非 optimal 槽数',ylabel='回合数',ylim=(0,350))
    ax=axes[1,1]
    slots=d.groupby('slot').failure.mean()*100
    ax.bar(slots.index,slots.values,color=['#A0ABB8']*20+['#D17C00']*20)
    ax.set(title='失败集中于后20槽（占全部失败的69.95%）',xlabel='槽序号（0–39）',ylabel='该槽非 optimal 比例 (%)')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('MHBPPO：训练仍有改善迹象，整回合可靠性尚未达到50%\n冻结至 update 6602 / 2026-10-20 09:29:53；训练窗口不是冻结策略评估',fontsize=13)
    fig.savefig(OUT/'episode_reliability.png',dpi=160)
    fig.savefig(OUT/'episode_reliability.svg')
    print(json.dumps({'passed':True,'checks':checks,'findings':findings,'matching_checks':len(matching_checks)},ensure_ascii=False))

if __name__=='__main__':main()

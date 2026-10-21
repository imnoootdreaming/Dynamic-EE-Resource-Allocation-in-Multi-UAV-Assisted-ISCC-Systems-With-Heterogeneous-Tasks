# -*- coding: utf-8 -*-
"""Summarize latest frozen learning, reliability and plateau evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/'mhbppo_latest_trend_20261020_2106'

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    a=json.loads((OUT/'audit.json').read_text(encoding='utf8'))
    s=json.loads((OUT/'step_summary.json').read_text(encoding='utf8'))
    prev=json.loads((PREVIOUS/'results.json').read_text(encoding='utf8'))
    assert a['passed'] and all(s['checks'].values())
    e=pd.read_csv(OUT/'episodes.csv');earlier=pd.read_csv(PREVIOUS/'episodes.csv')
    assert earlier.equals(e[e['update']<=prev['freeze']['last_update']].reset_index(drop=True))
    n=r['freeze']['last_update'];latest=r['windows']['last500'];prior=r['windows']['previous500'];short=r['windows']['last100'];old=prev['windows']['last500']
    def window(lo,hi):
        x=e[e['update'].between(lo,hi)]
        return {'range':[lo,hi],'episodes':len(x),'completion':float(x.all_completion.mean()*100),
                'optimal':float(x.all_optimal.mean()*100),'failures':float(x.optimal_failures.mean())}
    extended={name:window(lo,hi) for name,lo,hi in [('previous1000',n-1999,n-1000),('last1000',n-999,n),('previous250',n-499,n-250),('last250',n-249,n)]}
    (OUT/'extended_windows.json').write_text(json.dumps(extended,ensure_ascii=False,indent=2),encoding='utf8')
    u=e.groupby('update')[['all_optimal','all_completion','optimal_failures']].mean()
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,label,color in [('all_optimal','40槽全部 optimal','#087F8C'),('all_completion','40槽全部 completion','#D17C00')]:
        axs[0,0].plot(u.index,u[col].rolling(500).mean()*100,label=label,color=color,lw=2)
        axs[0,1].plot(u.index,u[col].rolling(100).mean()*100,label=label,color=color,lw=1.5)
    for ax in axs[0]:
        ax.axhline(50,color='#B64949',ls='--',lw=1.2,label='50%目标')
        ax.legend(loc='lower right',fontsize=8);ax.set(xlabel='update',ylabel='回合比例 (%)')
    axs[0,0].set(title='500更新窗口：连续两个完整窗口超过50%',xlim=(6000,n+80),ylim=(0,70))
    axs[0,0].axvline(9566,color='#888',ls=':',lw=1)
    axs[0,1].set(title='100更新窗口：仍有波动，近期略回落',xlim=(9500,n+80),ylim=(30,75))
    axs[1,0].plot(u.index,u.optimal_failures.rolling(500).mean(),color='#5B5EA6',lw=2)
    axs[1,0].set(title='平均失败槽数：近期约0.61，改善趋缓',xlim=(9000,n+80),ylim=(0,1.1),xlabel='update',ylabel='每回合非 optimal 槽数')
    blocks=[b for b in r['blocks500'] if b['range'][0]>=9501 and b['episodes']==1000]
    x=np.arange(len(blocks));labels=[f'{b["range"][0]}\n–{b["range"][1]}' for b in blocks]
    b1=axs[1,1].bar(x-.18,[b['all_completion_pct'] for b in blocks],width=.36,label='completion',color='#D17C00')
    b2=axs[1,1].bar(x+.18,[b['all_optimal_pct'] for b in blocks],width=.36,label='optimal',color='#087F8C')
    axs[1,1].bar_label(b1,fmt='%.1f',fontsize=9,padding=2);axs[1,1].bar_label(b2,fmt='%.1f',fontsize=9,padding=2)
    axs[1,1].axhline(50,color='#B64949',ls='--',lw=1)
    axs[1,1].set(xticks=x,xticklabels=labels,ylim=(40,65),title='固定500更新分块（每块1000回合）',xlabel='update区间',ylabel='全40槽回合比例 (%)')
    axs[1,1].legend(loc='upper left',fontsize=8)
    for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'MHBPPO：训练窗口已达到50%，近期出现平台波动迹象\n冻结至 update {n} / {r["freeze"]["last_update_time"]}；固定策略尚未独立验收',fontsize=12)
    fig.savefig(OUT/'latest_trend.png',dpi=160);fig.savefig(OUT/'latest_trend.svg');plt.close(fig)
    table='\n'.join(f'| {label} | {w["episodes"]} | {w["all_completion_pct"]:.1f}% | {w["all_optimal_pct"]:.1f}% | {w["mean_optimal_failures"]:.3f} |' for label,w in [(f'上次：9067–9566',old),(f'前500：{n-999}–{n-500}',prior),(f'末500：{n-499}–{n}',latest),(f'末100：{n-99}–{n}',short)])
    blocktable='\n'.join(f'| {b["range"][0]}–{b["range"][1]} | {b["episodes"]} | {b["all_completion_pct"]:.1f}% | {b["all_optimal_pct"]:.1f}% | {b["mean_optimal_failures"]:.3f} |' for b in r['blocks500'] if b['range'][0]>=9501)
    ci=r['recent_difference']['all_optimal']['25']['bootstrap95_pp']
    etable='\n'.join(f'| {k}：{w["range"][0]}–{w["range"][1]} | {w["episodes"]} | {w["completion"]:.2f}% | {w["optimal"]:.2f}% | {w["failures"]:.4f} |' for k,w in extended.items())
    t=a['timing']['last500'];bs=a['tensorboard'];counts=a['latest_solver_counts']
    report=f'''# 2026-10-21 最新训练检查

冻结至{r['freeze']['last_update_time']}，update {n}，{n*80}步、{n*2}条完整回合。对象为同一2026-10-19_09-44-15 run，42/0固定外生场景、同步2env、alpha0。

## 判断

1. 训练窗口层面的50%目标已经达到：最近两个互不重叠500-update窗口（各1000回合）全completion分别{prior['all_completion_pct']:.1f}%/{latest['all_completion_pct']:.1f}%，全optimal分别{prior['all_optimal_pct']:.1f}%/{latest['all_optimal_pct']:.1f}%。这满足U62提出的训练过程观察条件。
2. 相比第9566次更新仍有明显进步，但最新1000次update内平均失败数基本不变，全回合成功率略回落，表现为近期平台附近波动。尚不足以断言已完全收敛或发生显著退化；应更新上次“明显持续改善”的判断。
3. 优先转向固定checkpoint评价：保留到12000检查点，比较11000与12000的配套权重/normalization。先各200条独立动作seed筛选，再对选定模型用未参与筛选的至少1000条完整回合确认。当前日志不是固定模型的正式可靠性验收。

## 回合统计

| 窗口 | 回合数 | 全40槽completion | 全40槽optimal | 平均非optimal槽数 |
|---|---:|---:|---:|---:|
{table}

末500计数为全completion {latest['all_completion_n']}/1000、全optimal {latest['all_optimal_n']}/1000；两个worker全completion为{latest['per_worker']['0']['all_completion_pct']:.1f}%/{latest['per_worker']['1']['all_completion_pct']:.1f}%，全optimal为{latest['per_worker']['0']['all_optimal_pct']:.1f}%/{latest['per_worker']['1']['all_optimal_pct']:.1f}%。不是仅一个worker达标。两个worker共享场景，不代表两个独立场景或训练seed。

固定500分块如下；最后一行尚不足500次update，不能冒充完整1000回合窗口：

| 更新区间 | 回合数 | 全completion | 全optimal | 平均非optimal槽数 |
|---|---:|---:|---:|---:|
{blocktable}

补充较长和较短的非重叠窗口，避免仅挑一个端点：

| 窗口 | 回合数 | 全completion | 全optimal | 平均非optimal槽数 |
|---|---:|---:|---:|---:|
{etable}

## 是否仍在学习或已经退化

上次末500的每回合非optimal为{old['mean_optimal_failures']:.3f}，本次{latest['mean_optimal_failures']:.3f}，较长跨度仍改善；但本次相邻前后500分别{prior['mean_optimal_failures']:.3f}→{latest['mean_optimal_failures']:.3f}，几乎持平。

相邻两个500窗口逐槽optimal为{prior['slot_optimal_pct']:.4f}%→{latest['slot_optimal_pct']:.4f}%，reward {prior['reward']:.6f}→{latest['reward']:.6f}，Q {prior['Q']:.8f}→{latest['Q']:.8f}。这些平均指标均只有微小变化；全optimal却{prior['all_optimal_pct']:.1f}%→{latest['all_optimal_pct']:.1f}%，全completion {prior['all_completion_pct']:.1f}%→{latest['all_completion_pct']:.1f}%。说明减少平均失败数与把失败集中/分散到不同回合是不同问题。

按25-update移动块保留两个worker的bootstrap，全optimal差值95%描述区间[{ci[0]:.1f},{ci[1]:.1f}]个百分点，跨0；块长10/50同样跨0。没有充分证据把这次回落判成明确退化，也不宜再声称近期全回合率明显上升。这些区间只描述非平稳训练窗口，不是固定策略成功率的置信保证。

最新失败数分布：{latest['optimal_failure_hist']}。前500为{prior['optimal_failure_hist']}。零失败回合减少、仅1槽失败回合增多，同时部分多槽失败回合减少，因此总失败数近乎不变而整回合成功率下降。

![训练趋势]({(OUT/'latest_trend.png').as_posix()})

## 训练健康与失败诊断

- 配置、外生场景及30生产SHA与原run一致；权重版本/每轮80步/两个worker/checksum、两份训练CSV一致。与U62重叠的19132条episode逐行相同。
- BS七个训练标量全程完整且有限；未发现WARNING/ERROR。学习率保持3e−5下限。相邻前后500 critic loss均值{bs['BS/Critic_Loss']['windows']['previous500']['mean']:.4f}→{bs['BS/Critic_Loss']['windows']['last500']['mean']:.4f}，样本KL均值{bs['BS/KL_Divergence']['windows']['last500']['mean']:.6f}；不以单个loss或接近0的经验KL证明收敛。
- 最新40000槽solver计数：{counts}。解析感知条件失败{s['sensing_failure_n']}槽；失败UAV数分布{s['failed_uav_hist']}，感知裕量中位数{s['failed_margin_median']:.4f}。原始solver与解析重分类来源：{s['status_sources']}。同一证书生成的标签不当作独立双证据。
- 后20槽失败率{s['failure_slot_groups']['20_39']['failure_rate_pct']:.4f}%，占全部失败{s['failure_slot_groups']['20_39']['share_of_failures_pct']:.2f}%；U62相应失败率2.715%。
- 当前功率候选感知匹配存在比例{s['rematching']['current_power_matching']['pct']:.2f}%，裕量2匹配{s['rematching']['current_power_margin2_matching']['pct']:.2f}%。只是固定日志状态的必要条件反事实，不是重求解后的完整任务修复率。
- 恢复波束约束通过{s['beam_pass_pct']:.4f}%，加外层动作{s['beam_and_outer_pass_pct']:.4f}%；仍有{s['legacy_success_beam_failed_n']}个legacy success槽未通过恢复波束检查。因此日志全optimal/completion超过50%，不能自动解释成原约束下物理任务全部完成超过50%。

## 后续行动建议

若主要目标就是50%，现在应优先正式验收已得到的策略，不必为了观察曲线无限追加训练。固定11000/12000检查点及归一化统计、使用分离的动作采样seed；固定场景的确定性均值/argmax重复同一轨迹，不能当成独立样本。如果要求“有统计把握至少50%”，应预先确定置信区间与下界要求，而不是只凭样本均值。

若希望进一步稳定在60%或更高，先确认最近平台是否持续：可用12001–12500的完整窗口与前一个不重叠窗口比较；若均值失败数和全回合率继续无改善，再做U61提出的CU匹配/功率条件化、最差UAV裕量反馈、整回合风险目标的单因素实验。现有数据并不证明某项改法一定有效。比起同时改网络、熵、学习率、奖励，应保留当前基线供对照。

## 耗时

末500平均墙钟{t['cycle_seconds']['mean']:.3f}s/轮，采集等待{t['collection_wait_seconds']['mean']:.3f}s、PPO{t['update_seconds']['mean']:.3f}s（中位数{t['update_seconds']['median']:.3f}s）、日志{t['logging_seconds']['mean']:.3f}s。耗时证据与学习/收敛证据分别判断，不从单轮变慢推导策略不再学习。

## 验收产物与范围

results/audit/step_summary/extended_windows/final_validation JSON、完整episodes.csv、最新40000槽latest_steps.csv、失败诊断与原始JSONL、BS与离散策略标量、PNG/SVG、冻结CSV/text/hyper及SHA清单均保存在本目录。复用U61/U62脚本仅改变输出目录，旧证据不覆盖。本次只读训练数据与源码；未执行新策略评估/solver、未修改生产网络/奖励/参数、未启动或停止训练。
'''
    (OUT/'report.md').write_text(report,encoding='utf8')
    final={'passed':True,'audit':a['checks'],'steps':s['checks'],'U62_19132_episodes_unchanged':True,
           'solver_calls':0,'optimizer_steps':0,'freeze':r['freeze']}
    (OUT/'final_validation.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'freeze_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'frozen').iterdir()},indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'extended':extended,'latest500':latest,'step_summary':s,'timing':t},ensure_ascii=False))

if __name__=='__main__':main()

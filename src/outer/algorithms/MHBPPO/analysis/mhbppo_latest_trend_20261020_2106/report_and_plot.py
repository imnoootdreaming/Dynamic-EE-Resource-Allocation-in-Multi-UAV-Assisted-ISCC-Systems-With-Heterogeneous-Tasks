# -*- coding: utf-8 -*-
"""Render frozen trend evidence and a concise Chinese analysis report."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'mhbppo_episode_reliability_20261020'

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    a=json.loads((OUT/'audit.json').read_text(encoding='utf8'))
    s=json.loads((OUT/'step_summary.json').read_text(encoding='utf8'))
    old=json.loads((OLD/'results.json').read_text(encoding='utf8'))
    n=r['freeze']['last_update'];e=pd.read_csv(OUT/'episodes.csv');eo=pd.read_csv(OLD/'episodes.csv')
    assert eo.equals(e[e['update']<=old['freeze']['last_update']].reset_index(drop=True))
    assert a['passed'] and all(s['checks'].values())
    u=e.groupby('update')[['all_optimal','all_completion','optimal_failures']].mean()
    d=pd.read_csv(OUT/'latest_steps.csv');slots=d.groupby('slot').failure.mean()*100
    oldslots=pd.read_csv(OLD/'by_slot.csv').set_index('slot').failure_pct
    latest=r['windows']['last500'];prior=r['windows']['previous500'];short=r['windows']['last100'];was=old['windows']['last500']
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    c1='#087F8C';c2='#D17C00'
    for col,label,color in [('all_optimal','40槽全 optimal',c1),('all_completion','40槽全 completion',c2)]:
        axs[0,0].plot(u.index,u[col].rolling(500).mean()*100,label=label,color=color,lw=2)
        axs[0,1].plot(u.index,u[col].rolling(100).mean()*100,label=label,color=color,lw=1.5)
    for ax in axs[0]:
        ax.axhline(50,ls='--',color='#B64949',lw=1.2,label='50%目标')
        ax.set(ylabel='回合比例 (%)',xlabel='update',ylim=(0,65));ax.legend(loc='upper left',fontsize=8)
    axs[0,0].axvline(6602,color='#888',ls=':',lw=1)
    axs[0,0].set(title='500更新窗口（1000回合）：仍在接近50%',xlim=(3000,n+100))
    axs[0,0].annotate(f'最新：{latest["all_optimal_pct"]:.1f}% / {latest["all_completion_pct"]:.1f}%',xy=(n,latest['all_optimal_pct']),xytext=(7000,57),arrowprops={'arrowstyle':'->','color':'#666'},fontsize=9)
    axs[0,1].set(title='100更新窗口（200回合）：短期越过50%',xlim=(6500,n+100))
    ax=axs[1,0]
    ax.plot(u.index,u.optimal_failures.rolling(500).mean(),color='#5B5EA6',lw=2)
    ax.set(xlim=(6000,n+100),ylim=(0,2.5),title='500更新窗口：每回合失败槽数继续下降',xlabel='update',ylabel='非 optimal 槽数')
    ax.annotate('上次 1.777',xy=(6602,1.777),xytext=(6100,2.1),arrowprops={'arrowstyle':'->','color':'#666'},fontsize=9)
    ax.annotate(f'本次 {latest["mean_optimal_failures"]:.3f}',xy=(n,latest['mean_optimal_failures']),xytext=(8500,.3),arrowprops={'arrowstyle':'->','color':'#666'},fontsize=9)
    ax=axs[1,1]
    ax.plot(oldslots.index,oldslots.values,color='#AAA',label='上次：6103–6602',lw=1.7)
    ax.plot(slots.index,slots.values,color=c1,label=f'本次：{n-499}–{n}',marker='.',lw=1.7)
    ax.set(title='逐槽非 optimal 比例：后半程也在改善',xlabel='槽序号（0–39）',ylabel='非 optimal (%)',ylim=(0,None))
    ax.legend(fontsize=8)
    for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'MHBPPO 最新训练：继续改善，短窗口已过50%，长窗口尚未稳定达标\n冻结至 update {n} / {r["freeze"]["last_update_time"]}；训练窗口不等同固定策略评价',fontsize=12)
    fig.savefig(OUT/'latest_trend.png',dpi=160);fig.savefig(OUT/'latest_trend.svg');plt.close(fig)
    table='\n'.join(f'| {b["range"][0]}–{b["range"][1]} | {b["all_completion_pct"]:.1f}% | {b["all_optimal_pct"]:.1f}% | {b["mean_optimal_failures"]:.3f} |' for b in r['blocks500'] if b['range'][0]>=6501 and b['episodes']==1000)
    compare='\n'.join(f'| {name} | {b["episodes"]} | {b["all_completion_pct"]:.1f}% | {b["all_optimal_pct"]:.1f}% | {b["mean_optimal_failures"]:.3f} |' for name,b in [('上次：6103–6602',was),(f'前500：{n-999}–{n-500}',prior),(f'末500：{n-499}–{n}',latest),(f'末100：{n-99}–{n}',short)])
    ci=r['recent_difference']['all_optimal']['25']['bootstrap95_pp']
    t=a['timing'];bs=a['tensorboard'];risk=s['rematching']
    text=f'''# 最新训练趋势复核：是否仍学习、是否达到整回合50%

冻结：{r['freeze']['last_update_time']}，第{n}次update，{n*80}环境步、{n*2}条完整40槽回合。与上次第6602次相差{n-6602}次update；同一run、42/0固定场景、同步2环境、alpha0。

## 结论

当前比上次更有依据判断仍在有效改善，没有形成明确平台期。建议保留现有网络、奖励和参数，继续至10000检查，必要时按500更新延伸到10500、11000。短期全optimal和completion均越过50%，但最新1000条回合尚未达50%，更不能宣称冻结策略已稳定达到50%。原来“观察至8000”是复查节点，不是强制停止点；8000之后的实际改善支持继续。

## 主要数据

| 窗口 | 回合数 | 全40槽completion | 全40槽optimal | 平均非optimal槽数 |
|---|---:|---:|---:|---:|
{compare}

末500有{latest['all_completion_n']}条全completion、{latest['all_optimal_n']}条全optimal。两者距50%分别还差{50-latest['all_completion_pct']:.1f}、{50-latest['all_optimal_pct']:.1f}个百分点。末100分别{short['all_completion_n']}/200、{short['all_optimal_n']}/200。全optimal是日志solver_status_kind口径；completion还含外层动作条件，两者均不是原非凸问题全局最优的证明。

按固定、互不重叠500次更新分块，每行1000回合：

| update | 全completion | 全optimal | 平均非optimal槽数 |
|---|---:|---:|---:|
{table}

6501–8000一度缓慢改善，8001–8500明显上升，8501–9000有回落，9001–9500继续上升。不能把某个回落窗口单独判成收敛，也不能要求训练指标逐窗口单调。

相邻两个末500窗口：逐槽optimal {prior['slot_optimal_pct']:.4f}%→{latest['slot_optimal_pct']:.4f}%，平均reward {prior['reward']:.6f}→{latest['reward']:.6f}，Q {prior['Q']:.8f}→{latest['Q']:.8f}。全optimal提升8.3个百分点，按update保留两个worker、25-update移动块bootstrap的95%描述性差值区间为[{ci[0]:.1f}, {ci[1]:.1f}]个百分点；块长10/50也为正。比上次+2.1个百分点且区间跨0更有说服力，但它仍是非平稳训练窗口的描述，不能当成固定策略成功率的置信保证。

两个worker的末500全optimal分别{latest['per_worker']['0']['all_optimal_pct']:.1f}%、{latest['per_worker']['1']['all_optimal_pct']:.1f}%；前500分别{prior['per_worker']['0']['all_optimal_pct']:.1f}%、{prior['per_worker']['1']['all_optimal_pct']:.1f}%。改善不是单个worker独自贡献，两个worker共享外生场景，不能当作两个独立场景/训练seed。

![最新趋势]({(OUT/'latest_trend.png').as_posix()})

## 训练健康和策略变化

- CSV、逐worker文本、最新40000槽全部对齐；每update80步、worker版本update−1、checksum正常。配置、共享场景与上次相同，30个运行源码SHA与当前一致，未发现WARNING/ERROR文本。
- BS的7个训练标量覆盖1–{n}，均无NaN/Inf。相邻末500的critic loss均值{bs['BS/Critic_Loss']['windows']['previous500']['mean']:.4f}→{bs['BS/Critic_Loss']['windows']['last500']['mean']:.4f}，没有数值发散证据；reward归一化和目标分布在变化，不能把loss下降单独当作收敛证明。
- Actor loss较上次上升不等同策略变差：它是含熵项、归一化advantage与clip的动态代理目标。判学习效果应优先看回合成功率和失败数，源码见MHBPPO_agent.py的update。
- 学习率已到3e−5下限，之后不会自动变成0；日志KL监测值末500均值约0.000421。这是样本均值估计，部分更新为负，不应把接近0解释为精确KL为0或参数完全不动。
- 离散匹配平均最大类别概率：上次91.52%→本次92.28%，平均每头熵0.2191→0.1957；位移头条件标准差0.24695→0.22531。CU功率0.27722→0.27990、卸载时长0.28067→0.28493，并未全面收缩。各窗口访问状态不同，分布统计只表明策略/访问状态发生变化，不证明哪一头造成收益，也不支持“必须先整体降熵才能学习”。

## 最新失败模式

40000槽中39180 optimal、820 infeasible，unknown/error均0。所有820次均违反当前动作的感知必要条件；原始状态分解见step_summary.json（部分由raw unknown经同一解析证书重分类，不能当作独立双证据）。

- 感知失败UAV数量分布：{s['failed_uav_hist']}（808次仅1架、12次2架）；最差裕量中位数{s['failed_margin_median']:.4f}。820次中787次由raw unknown经解析证书判infeasible，33次为原始solver infeasible。
- 后20槽有{s['failure_slot_groups']['20_39']['n']}次失败，占全部失败{s['failure_slot_groups']['20_39']['share_of_failures_pct']:.2f}%。该段逐槽失败率{s['failure_slot_groups']['20_39']['failure_rate_pct']:.3f}%，上次为6.215%；因此后半程也在改善，不能把全部增益归于简单早期槽。
- 前4槽失败18→36次、失败率0.45%→0.90%，有局部回退；数量仍较少，不据此宣称灾难性遗忘，但不能说每个时段都改善。
- 实际功率下有{risk['current_power_matching']['n']}/{len(d[d.failure])}个失败状态存在满足感知必要条件的替代一对一匹配（{risk['current_power_matching']['pct']:.2f}%）；要求裕量≥2仍有{risk['current_power_margin2_matching']['n']}个。此处是候选条件检查，不是完整任务反事实求解。
- 末1000回合非optimal槽数分布为{latest['optimal_failure_hist']}。仅1槽失败仍有331条；保持其他结果不变时，只需再把23条失败回合变成全optimal即可让该窗口点估计到50%，但不是实际干预收益承诺。
- 恢复波束原约束通过率{s['beam_pass_pct']:.4f}%，加外层动作{s['beam_and_outer_pass_pct']:.4f}%；仍有{s['legacy_success_beam_failed_n']}个legacy success槽未通过恢复波束约束检查。原物理约束判据差异仍存在，不能将solver口径达标自动称为物理任务达标。

## 后续判断与执行建议

当前优先级是延续基线并验收，不是立即改网络或奖励。先看10000附近完整500窗口，再按500更新观察10500/11000；如继续改善则有继续训练价值。若连续两个不重叠500窗口全回合率和失败数都无实质改善，再启动U61的匹配反事实/裕量反馈等单因素消融。

训练观察可要求连续两个完整500窗口的全completion和全optimal均超过50%，以减少短窗口偶然越线的影响；这仍不替代固定模型验收。正式判断应固定checkpoint和配套normalization、停止更新统计量，在相同外生场景下改变独立动作采样seed，评价至少1000条完整回合并报告区间。固定场景的确定性均值/argmax重复同一轨迹，不能当成1000条独立试验。若要求“有统计把握至少50%”，应以预设置信下界为准，不能仅靠500/1000点估计。

当前尚未实施的功率条件化匹配、最差UAV裕量、整回合风险目标仍是后续方案；最新基线已自然取得明显进展，不应把这些未实施方案的潜在收益冒认为已经发生。

## 耗时补充

末500平均墙钟{t['last500']['cycle_seconds']['mean']:.3f}s/轮（前500 {t['previous500']['cycle_seconds']['mean']:.3f}s）；末100增至{t['last100']['cycle_seconds']['mean']:.3f}s/轮。末500采集等待{t['last500']['collection_wait_seconds']['mean']:.3f}s，PPO均值{t['last500']['update_seconds']['mean']:.3f}s、但中位数{t['last500']['update_seconds']['median']:.3f}s；少数PPO记录达22.16s，末100明显受长尾拖慢。只能确认近期变慢与长尾，缺少当时系统CPU/I/O/调度记录，不能把这些异常直接归因于2env、MOSEK或算法不收敛。

## 证据与边界

本次冻结和验证记录见results.json、audit.json、step_summary.json、final_validation.json；详细表见episodes.csv、latest_steps.csv、failed_step_details.csv、ppo_scalars.csv、discrete_policy.csv。所有与上次重叠的13204条episode逐行完全一致，未覆盖U61产物。本次未进行新策略评估、未调用solver、未改变生产源码/参数或操作训练进程。配置与源文件一致性支持可比，不构成多随机种子泛化结论。
'''
    (OUT/'report.md').write_text(text,encoding='utf8')
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'frozen').iterdir()}
    (OUT/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    final={'passed':True,'core':a['checks'],'steps':s['checks'],'U61_all_13204_episodes_unchanged':True,
           'old_frozen_results_sha256':hashlib.sha256((OLD/'results.json').read_bytes()).hexdigest(),
           'solver_calls':0,'optimizer_steps':0,'freeze':r['freeze']}
    (OUT/'final_validation.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'report':str(OUT/'report.md'),'all_optimal_pct':latest['all_optimal_pct'],'all_completion_pct':latest['all_completion_pct']},ensure_ascii=False))

if __name__=='__main__':main()

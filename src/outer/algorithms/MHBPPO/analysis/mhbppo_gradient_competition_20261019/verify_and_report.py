# -*- coding: utf-8 -*-
"""Independent vector arithmetic, provenance checks, and final report/figure."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'gradient_results.json').read_text(encoding='utf-8'))
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
vec=np.load(OUT/'gradient_vectors.npz',allow_pickle=False)
batch=np.load(OUT/'reconstructed_batch.npz',allow_pickle=False)
state=torch.load(OUT/'source_frozen/mhbppo_actor.pth',map_location='cpu',weights_only=True)
positions={};offset=0
for name,tensor in state.items():
    positions[name]=(offset,offset+tensor.numel());offset+=tensor.numel()
group_indices={g:np.concatenate([np.arange(*positions[n]) for n in names]) for g,names in d['parameter_groups'].items()}
checks={'artifact_sha_matches':all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in d['artifact_sha256'].items()),
    'frozen_source_unchanged':all(hashlib.sha256((OUT/'source_frozen'/p).read_bytes()).hexdigest()==h for p,h in protocol['source_hashes'].items()),
    'production_source_matches':all(protocol['production_source_matches'].values()),
    'no_solver_no_optimizer':d['reconstruction']['solver_calls']==d['reconstruction']['optimizer_steps']==0,
    'direct_autograd_vs_affine_max_relative_error':max(x['gradient_relative_l2_error'] for x in d['direct_affine_checks']),
    'decomposition_checks':{},'numeric_checks':{}}
metadata=json.loads((OUT/'actions_update5001.jsonl').open(encoding='utf-8').readline())
checks['production_source_matches']=all(hashlib.sha256((OUT.parents[5]/'src/outer'/name).resolve().read_bytes()).hexdigest()==expected
    for name,expected in metadata['source_sha256'].items())
checks['analysis_source_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    for name in ('prepare_inputs.py','compute_gradients.py','verify_and_report.py')}
checks['reconstruction_helper_sha256']=hashlib.sha256((OUT.parents[5]/'src/outer/algorithms/MHBPPO/generate_beam_pretrain_dataset.py').read_bytes()).hexdigest()
checks['hyperparameters_sha256']=hashlib.sha256((OUT/'hyperparameters.json').read_bytes()).hexdigest()
for label,parts in [('workers',('worker0_contribution','worker1_contribution')),
                    ('slots',('slots0_7_contribution','slots8_39_contribution'))]:
    checks['decomposition_checks'][label]={}
    for suffix in ('task_a','task_b','entropy_beam','entropy_other'):
        combined=vec[f'{parts[0]}__{suffix}']+vec[f'{parts[1]}__{suffix}']
        original=vec[f'all80__{suffix}']
        checks['decomposition_checks'][label][suffix]=float(np.linalg.norm(combined-original)/max(np.linalg.norm(original),1e-30))
a=batch['gae_a'];b=batch['gae_b'];a=a-a.mean();b=b-b.mean()
lo=d['reward_normalizer']['unknown_std_lower_bound'];hi=d['reward_normalizer']['unknown_std_upper_bound']
sigmas=np.linspace(lo,hi,301)
plot_values={}
for group in ('beam_heads','shared_trunk'):
    indices=group_indices[group]
    ga=vec['all80__task_a'][indices];gb=vec['all80__task_b'][indices];gh=vec['all80__entropy_beam'][indices]
    result=d['results']['all80'][group]
    ratios=[];cosines=[]
    for sigma in sigmas:
        q=sigma+1e-8
        denom=np.std(a+q*b,ddof=1)+q*1e-5
        task=(ga+q*gb)/denom
        ratios.append(float(np.linalg.norm(gh)/np.linalg.norm(task)))
        cosines.append(float(gh@task/(np.linalg.norm(gh)*np.linalg.norm(task))))
    bounds=result['norm_ratio_analytic_bounds'];cmin=result['cosine_range']['min']['cosine'];cmax=result['cosine_range']['max']['cosine']
    assert min(ratios)>=bounds['lower']-1e-10 and max(ratios)<=bounds['upper']+1e-10
    assert min(cosines)>=cmin-1e-10 and max(cosines)<=cmax+1e-10
    projection_upper=bounds['upper']*max(0.,-cmin)
    checks['numeric_checks'][group]={'direct_vector_ratio_min':min(ratios),'direct_vector_ratio_max':max(ratios),
        'ratio_within_analytic_bounds':True,'cos_within_analytic_bounds':True,
        'opposing_projection_fraction_conservative_upper_bound':projection_upper,
        'angle_degrees_min':math.degrees(math.acos(cmax)),'angle_degrees_max':math.degrees(math.acos(cmin))}
    plot_values[group]=(ratios,cosines)
checks['passed']=all(checks[k] for k in ('artifact_sha_matches','frozen_source_unchanged','production_source_matches','no_solver_no_optimizer'))
checks['passed'] &= checks['direct_autograd_vs_affine_max_relative_error']<3e-5
checks['passed'] &= max(v for x in checks['decomposition_checks'].values() for v in x.values())<3e-6
assert checks['passed'],checks
(OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(2,1,figsize=(8.6,6),sharex=True,constrained_layout=True)
for group,color,label in [('beam_heads','#0072B2','Beam output heads'),('shared_trunk','#D55E00','Shared trunk')]:
    ratios,cosines=plot_values[group]
    axes[0].plot(sigmas,np.array(ratios)*100,color=color,label=label)
    axes[1].plot(sigmas,cosines,color=color,label=label)
axes[0].set_ylabel('Beam entropy / PPO\ngradient norm (%)')
axes[0].legend()
axes[1].axhline(0,color='gray',lw=1,ls='--')
axes[1].set_ylabel('Gradient cosine\nnegative = opposing')
axes[1].set_xlabel('Unknown historical reward standard deviation: sensitivity interval')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Checkpoint 5000 + logged update 5001 (80 samples)\nInitial full-batch gradients; no Adam steps; actual reward std is unavailable',fontsize=11)
fig.savefig(OUT/'gradient_sensitivity.png',dpi=170)
fig.savefig(OUT/'gradient_sensitivity.svg')
plt.close(fig)

beam=d['results']['all80']['beam_heads'];trunk=d['results']['all80']['shared_trunk']
out=['# U58：beam熵梯度与PPO任务梯度的实际离线计算','',
'**结论：本次可重建批次中，没有看到beam熵梯度强烈抵消PPO任务梯度的证据。beam头上熵梯度范数仅约为任务梯度的3.68%–4.65%，夹角近正交；这不支持将“降beam熵”继续视为有实证依据的首要修复。当前仍无法精确还原当时实际的GAE及完整Adam更新，下面完整保留这一限制。**','',
'## 1. 计算对象与缺失项','',
'- 使用该run的`iter_5000` actor/critic/配套state norm，以及紧接着的update5001、worker0/1各40槽的真实历史动作与奖励。不是新采样，不是用5000权重解释5804权重的样本。',
'- 原日志没有保存raw/normalized states、old log-prob和GAE；依据同源物理参数、固定场景、时隙和日志位置，重建80个S及S_next。源码30/30 SHA一致、actor/norm/scene签名一致；重算全部策略头mean/std/concentration/probs与采集记录最大绝对误差1.91e−6，连续轨迹相邻状态一致。没有原始状态字节副本可作直接逐字节对照。',
'- reward-normalizer未保存在checkpoint中。5000批的原始均值仍在CSV，因此历史reward mean可重建，历史std/m2不唯一，不能随意指定后冒充真实值。',
'- 计算的是更新入口、theta=theta_old处80条样本的完整batch loss梯度；此时两个ratio理论上均为1，clip局部未激活。不是历史随机minibatch内的梯度，也不是5个epoch/15次Adam的更新轨迹。Adam状态与minibatch排列未保存。','',
'## 2. 如何处理未知的奖励标准差','',
f'历史reward mean约为 {d["reward_normalizer"]["mean_reconstructed_from_400000_rewards_batch_means"]:.12f}。按当前非负能耗和I=4动作范围，奖励有保守界[-9,1]；总方差不小于5000个等大小batch均值的方差，且对有界R满足Var(R)≤(1−mu)(mu+9)。由此得到std的保守区间 **[{lo:.9f}, {hi:.9f}]**。这不是对实际std的估计。','',
'固定critic时，未标准化GAE对奖励std具有可分离结构：`A(sigma)=a/(sigma+1e-8)+b`。a来自原始奖励减历史均值，b来自critic TD项，均按两个40槽边界分别递推。将全batch advantage按原代码标准化后，初始任务梯度为：','',
'`g_task(sigma) = [g_a + (sigma+1e-8) g_b] / [std_unbiased(a+(sigma+1e-8)b) + (sigma+1e-8)*1e-5]`。','',
'因此无需猜测实际std即可求夹角的区间极值，并通过有理二次函数驻点给范数比提供覆盖整个区间的解析上下界。另在4个std点按原float32 TD/GAE代码直接反向传播，核对该分解。','',
'## 3. 计算结果','',
'这里g_task是原连续组+离散组surrogate的梯度；g_H是**原系数已经乘入的**`grad(-0.05*H_beam)`。均为loss对参数的欧氏梯度，不是Adam预条件后的步长。','',
'| 参数组 | 熵梯度范数 | 熵/任务梯度范数 | 梯度余弦 |',
'|---|---:|---:|---:|']
for name,r in [('beam输出头（20640参数）',beam),('共享干网络',trunk)]:
    bounds=r['norm_ratio_analytic_bounds']
    out.append(f'| {name} | {r["beam_entropy_norm"]:.8f} | {bounds["lower"]*100:.3f}%–{bounds["upper"]*100:.3f}% | {r["cosine_range"]["min"]["cosine"]:.5f}–{r["cosine_range"]["max"]["cosine"]:.5f} |')
out += ['',
f'beam头夹角约 {checks["numeric_checks"]["beam_heads"]["angle_degrees_min"]:.2f}°–{checks["numeric_checks"]["beam_heads"]["angle_degrees_max"]:.2f}°。负cos才表示抵消，且当前未知std使符号不能唯一确定。即使取最保守的范数比与负cos上界，熵在任务梯度方向上的反向投影也不超过任务梯度范数的 **{checks["numeric_checks"]["beam_heads"]["opposing_projection_fraction_conservative_upper_bound"]*100:.3f}%**。共享干上该batch的beam熵梯度与任务梯度cos始终为正，没有反向抵消。','',
'worker0/1和前8/后32槽的分量也单独保存（保持全80条的GAE归一化与mean权重，未把子组冒充独立训练批）。各子组结果同样不能代表后续更新中的持续作用。','',
f'![缺失奖励std下的敏感性分析]({(OUT/"gradient_sensitivity.png").as_posix()})','',
'## 4. 能据此改变什么判断','',
'1. 不应继续将“熵太强导致方向学不出”列为有证据支持的首要原因。这批数据没有提供这种支持；降熵可以保留为单因素消融，应先扩大实际minibatch/epoch的诊断覆盖。在线物理辅助、方向信用分配和后段目标跟踪依然是待验证候选，不能因为本次未见熵强烈抵消就推定其他方法已经有效。',
'2. 不能反过来断言熵在长期训练无作用。有限样本任务梯度含噪声，小熵梯度可长期累积；clip后的后续epoch、Adam逐坐标缩放和共享特征漂移也可能改变相对作用。当前数据不能精确计算这些路径。',
'3. 如需精确评价实际训练路径，应在更新时记录reward-normalizer mean/std/count及GAE，或直接在内存中对同一个minibatch记录分项梯度范数/内积；跨epoch记录初期/末期及clip比例。若要离线完整复现，还需完整optimizer/RNG和minibatch索引。只记录scalar total loss或entropy值不够。',
'4. 本次没有实施熵修改、辅助学习或新PPO更新，没有执行env.step/reward/solver，没有操作用户训练进程。','',
'## 5. 数值与复现验收','',
f'- 直接float32反向传播与GAE分解的梯度相对L2误差最大 **{checks["direct_autograd_vs_affine_max_relative_error"]:.3e}**。',
'- 独立从梯度向量复算301个std点，全部落在解析夹角/范数比范围；worker与slot分量相加复现全batch梯度，误差低于3e−6。',
'- 固定checkpoint文件SHA、模型参数、source及归一化绑定均通过；未执行optimizer.step。',
'- protocol.json、冻结80条原始日志/5001行CSV/4个checkpoint文件、reconstructed_batch.npz、gradient_vectors.npz、gradient_results.json、validation.json和PNG/SVG均保留。复现依次运行prepare_inputs.py（依赖尚保留的源日志）、compute_gradients.py、verify_and_report.py；冻结输入已保留后，后两步不再依赖滚动日志。',
'- 准备阶段曾因冻结目录缺父级hyperparameters而拒错，补原字节副本后成功；独立报告脚本曾有UTF-8加载错误，补编码声明后验收通过。初版区间扫描按完整向量逐点评估，后改等价Gram标量式并重新核验。上述均未改生产源码或用失败启动结果代替计算。']
(OUT/'report.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print(json.dumps({'passed':checks['passed'],'beam_opposing_projection_upper':checks['numeric_checks']['beam_heads']['opposing_projection_fraction_conservative_upper_bound']},indent=2))

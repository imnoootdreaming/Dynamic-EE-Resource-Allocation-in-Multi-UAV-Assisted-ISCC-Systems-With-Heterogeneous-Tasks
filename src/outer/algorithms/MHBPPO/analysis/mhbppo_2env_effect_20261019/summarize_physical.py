"""Rebuild compact statistics/report from verified NPZ, without rescanning large logs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extract_physical", OUT / "extract_physical.py")
extract = importlib.util.module_from_spec(spec); spec.loader.exec_module(extract)
metadata = json.loads((OUT / "physical_metadata.json").read_text(encoding="utf-8"))
extract.PARAMS = metadata["inner_parameter_source"]
arrays = np.load(OUT / "physical_metrics.npz")
r, u, a = arrays["rows"], arrays["uav"], arrays["action"]
result = json.loads((OUT / "physical_summary.json").read_text(encoding="utf-8"))
for key, window in result["windows"].items():
    result["windows"][key] = extract.summarize(r, u, a, window["lo"], window["hi"])
for name in ("slot_groups_last500", "broad_slot_groups_last500"):
    result[name] = [extract.summarize(r, u, a, x["lo"], x["hi"], x["slots"]) for x in result[name]]
for worker in (0, 1):
    mask = r[:, 1] == worker
    result["last500_per_worker"][str(worker)] = extract.summarize(r[mask], u[mask], a[mask], 5305, 5804)
root = OUT.parents[5]
source = []
for name, expected in metadata["source_sha256"].items():
    path = (root / "src/outer" / name).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    source.append({"path": name, "expected_sha256": expected, "current_sha256": actual, "matches": actual == expected})
result["source_provenance"] = {"recorded_files": len(source), "current_match_count": sum(x["matches"] for x in source), "files": source}
result["compact_npz_sha256"] = hashlib.sha256((OUT / "physical_metrics.npz").read_bytes()).hexdigest()
extract.save("physical_summary.json", result)

p, q, z = result["windows"]["previous500"], result["windows"]["last500"], result["windows"]["last100"]
lines = ["# 当前 N2 训练的物理瓶颈证据", "",
"结论：当前完整日志窗口显示成功仍集中在 episode 开头，感知必要条件是失败的主要直接障碍；接收波束平均目标增益仍接近随机方向基准。CU 匹配、功率和轨迹仍有空间，但下面的替代匹配/降功率诊断不能证明完整任务可行。", "",
"## 证据边界", "",
"- CSV 冻结截至 update 5804；本报告仅详细抽取 4805–5804，1000 个完整更新、2 个 worker × 40 槽/更新，共 80000 条 env 和 80000 条 action，逐条对齐 actor/norm/scene 身份。",
"- 4805–5000 来自 part005，5001–5804 来自活动 JSONL；按打开文件句柄的实际长度做边界与 byte bisection，避免 PowerShell 目录项大小/mtime 缓存。活动文件实际持续写入，不能据目录项声称 5001 后停写。",
f"- 所有 1000 更新的 optimal count、Q、base reward、actual reward 与冻结 CSV 一致；M 公式、Q 与 reward 恒等式复算误差均 <1e-12。日志记录的 {len(source)} 个源码 SHA 与当前源码匹配 {sum(x['matches'] for x in source)} 个。",
"- 未运行策略、训练、solver；未修改生产数据、源码或参数。", "", "## 相邻窗口", "",
"| 指标 | 4805–5304 | 5305–5804 |", "|---|---:|---:|"]
for label, key, scale in [("optimal %","optimal_percent",1),("completion %","completion_percent",1),("Q","Q",1),
                        ("联合必要感知通过 %","sensing_pass_percent",1),("后 32 槽 optimal %","last32_optimal_percent",1),
                        ("执行目标 gain","gain_mean",1),("目标距离 m","distance_mean",1),("接收 beam Beta 坐标 std","beam_std_mean",1),
                        ("全部 CU 平均功率 W","avg_cu_power",1),("实际匹配 CU 平均功率 W","matched_power",1)]:
    lines.append(f"| {label} | {p[key]*scale:.6f} | {q[key]*scale:.6f} |")
lines += ["", "## 末 500 更新的时隙分解", "",
"| 槽 | 样本 | optimal % | 必要感知通过 % | Q | gain | 目标距离 m | 最低功率也无感知匹配 % |",
"|---|---:|---:|---:|---:|---:|---:|---:|"]
for x in result["slot_groups_last500"]:
    lines.append(f"| {x['slots'][0]}–{x['slots'][1]} | {x['records']} | {x['optimal_percent']:.4f} | {x['sensing_pass_percent']:.4f} | {x['Q']:.4f} | {x['gain_mean']:.6f} | {x['distance_mean']:.2f} | {x['no_min_matching_percent']:.4f} |")
lines += ["", f"末 500 更新中，前 8 槽占全部 optimal 的 {q['first8_share_of_optimal_percent']:.4f}%；{q['failed_steps_rejected_by_sensing_percent']:.4f}% 的失败步已违反感知必要条件。必要感知通过但未 optimal 共 {q['sensing_pass_but_not_optimal_count']} 步；本窗口的数值相等不能证明感知条件普遍充分，也不能保证替换动作后的完整求解成功。", "", "## 感知与 CU 干扰", "",
"固定外层几何、接收 beam、匹配与 CU 功率时，M_i=Pmax·λmax(G_i)/(ε·Γ_i)。任一 M_i<1−tol 即不可能同时满足感知最大功率和 SINR，属于固定动作的充分不可行证据；M_i≥1 仅通过必要条件。", "",
"当前感知通道为秩一形式 A=c·aaᴴ，因此 gain=λmax(G)·d⁴/(ξ₀ρref N²)=|aᴴg|²/N∈[0,1]。具有零均值和坐标对称的随机单位方向满足 E[ggᴴ]=I/N，故 E[gain]=1/N=0.1。这个基准是代码公式的代数结果；gain≈0.1 说明没有平均目标定向增益突破，不能据此排除局部改进或抑制干扰的学习。", "",
f"末 500 的干扰/噪声中位数为 {q['interference_noise_ratio_median']:.2f}；仅作零干扰上界诊断时，联合必要感知通过率为 {q['noise_only_joint_sensing_pass_percent']:.4f}%。零干扰不是合法 CU 功率方案，实际功率下限约为 {0.1*extract.PARAMS['P_max_cu']:.8f} W。", "",
"| 固定几何和接收 beam 的诊断 | 末 500 所有步占比 % |", "|---|---:|",
f"| 实际动作联合必要感知通过 | {q['sensing_pass_percent']:.4f} |",
f"| 实际 CU 功率下存在一对一感知匹配 | {q['current_power_matching_exists_percent']:.4f} |",
f"| 所有 CU 降至合法下限后存在一对一感知匹配 | {q['min_power_matching_exists_percent']:.4f} |",
f"| 保留实际匹配，仅把其功率降至下限后必要感知通过 | {q['selected_edges_at_min_power_joint_pass_percent']:.4f} |",
f"| 即使最低功率也无一对一感知匹配（认证） | {q['no_min_matching_percent']:.4f} |", "",
"存在感知匹配仅说明 c05/c06 子系统可通过；降功率可能损害 CU 上行任务，不是建议把所有功率一律压到下限。没有最低功率感知匹配则说明固定几何/beam 下单改匹配和功率仍不足，必须涉及 beam 或轨迹。", "", "## 轨迹与工程耗时", "",
f"末 500 对当前目标的每步平均距离减少量 {q['progress_toward_current_target_m_mean']:.6f} m；平均速度 {q['avg_speed']:.6f} m/s。该距离包括移动前后针对当前目标的同槽比较，不能单独评价为下一目标提前移动的策略优劣。时隙表的距离与必要通过率提示应逐个四槽窗口检查目标跟踪，不能以全程速度均值宣称轨迹已经学成。", "",
f"末 500，必要感知失败 {q['sensing_failed_count']} 步，仍消耗 solver 累计 {q['solver_seconds_sensing_failed_sum']:.3f} worker·s，均值 {q['solver_seconds_sensing_failed_mean']:.6f} s/步，占全部求解秒数 {100*q['solver_seconds_sensing_failed_share']:.4f}%；通过必要感知的步累计 {q['solver_seconds_sensing_passed_sum']:.3f} worker·s，均值 {q['solver_seconds_sensing_passed_mean']:.6f} s/步。",
"该累计是两 worker 的求解时间之和，不能直接等同可节省墙钟。cccp_bridge.py 已计算 sensing 后仍无条件 run_pc3p，因而可考虑在严格 M<1−tol 认证时提前返回不可行，并保留奖励、诊断和 raw-status provenance 契约；M≥1 仍调用 solver。预筛收益及语义一致性需另行验证，不能把理论时间上限称为实测提速。", "", "## 修改建议的边界", "",
"优先针对在线 beam 的相干结构和保持机制设计单因素实验，而非重复一次预热：U53 已有 500 更新预热组合未保持收益的负结果。可测试训练期物理辅助期望目标、beam 专属探索强度或专属分支/KL 保持，但要同时报告实际采样 gain、Γ、M、后 32 槽 optimal 与同墙钟收益。不要只降低全体连续动作熵，也不要把执行动作改成解析接收器后称为原任务上的 PPO 提速。", "",
"CU 条件匹配/动作掩码、协同功率与轨迹目标跟踪可作为分开的消融。任何解析替代、辅助 loss、熵或奖励改变都会更改算法；本次只给证据和方案，未实施。", "", "## 可复算文件", "",
"extract_physical.py 扫描原始选段并生成 physical_metrics.npz；summarize_physical.py 仅从紧凑 NPZ 重建 physical_summary.json 与本报告。schema、选段 raw-line SHA、文件句柄长度、CSV 验证范围、来源 SHA 全部写入 JSON。", ""]
(OUT / "physical.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"last500": {key:q[key] for key in ("optimal_percent","Q","sensing_pass_percent","gain_mean","failed_steps_rejected_by_sensing_percent","no_min_matching_percent","current_power_matching_exists_percent","min_power_matching_exists_percent","last32_optimal_percent","solver_seconds_sensing_failed_share")},"broad_slots":[{key:x[key] for key in ("slots","optimal_percent","sensing_pass_percent","Q","gain_mean","distance_mean")} for x in result["broad_slot_groups_last500"]]},indent=2))

from pathlib import Path
import shutil,json,hashlib
R=Path.cwd(); O=R/'analysis/mosek_runs_20261010'; B=O/'before'; B.mkdir(exist_ok=True)
files=['src/inner/parameters.py','src/inner/fusion_pc3p.py','src/inner/pc3p.py','src/inner/gc3p.py','src/inner/main.py','src/inner/compare_pc3p_gc3p.py']
inputs=files+['src/inner/feasible_outer_samples.csv']+[str(p.relative_to(R)).replace('\\','/') for p in (R/'src/inner').glob('*.csv') if p.name!='feasible_outer_samples.csv']
for name in inputs:
 p=R/name;dest=B/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(O/'before_manifest.json').write_text(json.dumps({p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf8')
plan=R/'src/outer/algorithms/MHBPPO/MHBPPO_TASK_PLAN.md'
s=plan.read_text(encoding='utf8');s=s.replace('> 最近更新：2026-10-09','> 最近更新：2026-10-10')
s=s.replace('## 3. 当前下一步','## 3. 当前下一步\n\n**用户新增优先任务 S02（2026-10-10，执行中）：将 RelGap=1e-7 接入求解路径，实际运行 main.py 和 compare_pc3p_gc3p.py 生成 CSV；同样本基准1e-8对照，保留50候选和seed=1208。**\n\nS01旧证据已核对，适用于修改前源码与配置；源码/默认精度改变后的结果以S02为准。T02/T03诊断实现不变，旧端到端数值证据保留其旧容差作用域，不外推到新默认配置。本次不启动训练，完成后继续T04。')
s+='\n\n## S02：入口运行与 CSV 验收（2026-10-10）\n\n| ID | 状态 | 输入与范围 | 验收条件 |\n|---|---|---|---|\n| S02 | 执行中 | 当前30样本；新RelGap字段及两个入口明细；PC3P/GC3P同精度；原CSV和六个源码备份在analysis/mosek_runs_20261010/before | 两入口完整退出；绘图CSV与明细对齐30行；基准/加速能耗与耗时对照；原始约束诊断保留；输入CSV不变 |\n'
plan.write_text(s,encoding='utf8')
def edit(name,old,new,count=1):
 p=R/name;s=p.read_text(encoding='utf8');assert s.count(old)==count,(name,old[:80],s.count(old));p.write_text(s.replace(old,new),encoding='utf8')
edit(files[0],'    mosek_tol_feas: float = 1e-8','    mosek_tol_feas: float = 1e-8\n    # S01测试的加速候选：仅放宽最优性间隙，原始/对偶可行性容差不变。\n    mosek_tol_rel_gap: float = 1e-7')
edit(files[1],'    求解器容差与模型结构无关，首次建模型时设置一次即可，后续复用无需重复设置。','    求解器容差与模型结构无关；每次获取时刷新，允许缓存模型用于不同精度的对照。')
edit(files[1],'    return cached\n','    model = cached[0]\n    model.setSolverParam("intpntCoTolPfeas", ctx.mosek_tol_feas)\n    model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)\n    model.setSolverParam("intpntCoTolRelGap", ctx.mosek_tol_rel_gap)\n    return cached\n')
edit(files[1],'        model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)\n\n    try:', '        model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)\n        model.setSolverParam("intpntCoTolRelGap", ctx.mosek_tol_rel_gap)\n\n    try:')
for name in [files[2],files[3]]:
 edit(name,'        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": ctx.mosek_tol_feas,','        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": ctx.mosek_tol_feas,\n        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": ctx.mosek_tol_rel_gap,')
edit(files[3],'    model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)','    model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)\n    model.setSolverParam("intpntCoTolRelGap", ctx.mosek_tol_rel_gap)')
helper=R/'src/inner/solution_checks.py'
assert not helper.exists()
helper.write_text('''"""Read-only original constraint diagnostics for the inner CSV entry points."""
import importlib.util
from pathlib import Path
import sys

_ALIAS = "_inner_csv_feasibility_diagnostics"
if _ALIAS not in sys.modules:
    path = Path(__file__).resolve().parents[1] / "outer/reward/feasibility_diagnostics.py"
    spec = importlib.util.spec_from_file_location(_ALIAS, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_ALIAS] = module
_validator = sys.modules[_ALIAS].validate_original_solution

CHECK_FIELDS = ["covariance_passed", "recovered_beams_passed", "rank_passed",
                "physical_solution_verified", "max_normalized_violation", "failed_checks"]

def solution_checks(ctx, result):
    report = _validator(ctx.params, ctx.environment, result)
    covariance, recovered = report["covariance"], report["recovered_beams"]
    return dict(covariance_passed=int(covariance["passed"]),
                recovered_beams_passed=int(recovered["passed"]),
                rank_passed=int(report["rank"]["passed"]),
                physical_solution_verified=int(report["physical_solution_verified"]),
                max_normalized_violation=max(covariance.get("max_normalized_violation", float("nan")),
                                             recovered.get("max_normalized_violation", float("nan"))),
                failed_checks=";".join(["covariance:" + k for k in covariance["failed_checks"]]
                                       + ["recovered:" + k for k in recovered["failed_checks"]]
                                       + ([] if report["rank"]["passed"] else ["rank"])) )
''',encoding='utf8')
edit(files[4],'import csv\n','import argparse\nimport csv\n')
edit(files[4],'from outer_sampler import load_outer_samples, sample_random_feasible','from outer_sampler import load_outer_samples, sample_random_feasible\nfrom pc3p import compute_pure_energy\nfrom solution_checks import CHECK_FIELDS, solution_checks')
edit(files[4],'                         "status_kind", "elapsed_s"])','                         "status_kind", "elapsed_s", "mosek_rel_gap",\n                         "objective_p4", "pure_energy", "rank1_gap"] + CHECK_FIELDS)')
edit(files[4],'def main():\n    params = Parameters()','def main():\n    parser = argparse.ArgumentParser(description="PC3P 内层求解与 CSV 输出")\n    parser.add_argument("--mosek-rel-gap", type=float, default=None,\n                        help="覆盖相对最优性间隙容差；默认 Parameters 配置")\n    args = parser.parse_args()\n    params = Parameters()\n    if args.mosek_rel_gap is not None:\n        if not 0 < args.mosek_rel_gap <= 1:\n            parser.error("--mosek-rel-gap 必须在 (0, 1] 内")\n        params.mosek_tol_rel_gap = args.mosek_rel_gap')
edit(files[4],'    print("求解后端：{}".format(backend_name))','    print("求解后端：{}；MOSEK RelGap={:.1e}".format(backend_name, params.mosek_tol_rel_gap))')
edit(files[4],'        iteration_rows.append([index, index - 1, result["iterations"],','        checks = solution_checks(ctx, result)\n        pure_energy = (compute_pure_energy(ctx, result["W_sen_beam"], result["B_off_beam"],\n                                           result["f_uav_freq"], result["z_aux_rate"], result["D_cu_off"])\n                       if result["W_sen_beam"] is not None else float("nan"))\n        iteration_rows.append([index, index - 1, result["iterations"],')
edit(files[4],'                               "{:.6f}".format(elapsed)])','                               "{:.6f}".format(elapsed), params.mosek_tol_rel_gap,\n                               result["objective_p4"], pure_energy, result["rank1_gap"]]\n                              + [checks[k] for k in CHECK_FIELDS])')
edit(files[4],'    # save_iteration_counts_detail(CONVERGENCE_DETAIL_CSV, iteration_rows)','    save_iteration_counts_detail(CONVERGENCE_DETAIL_CSV, iteration_rows)')
edit(files[4],'    # print("各 case 的完整求解统计已保存到：{}".format(CONVERGENCE_DETAIL_CSV))','    print("各 case 的完整求解统计已保存到：{}".format(CONVERGENCE_DETAIL_CSV))')
edit(files[4],'# 目前无需保存 details 文件，main() 中的写出调用已注释掉，此处仅保留路径与写出函数备用。','# 明细包含能耗、精度配置和独立约束诊断，用于判断加速影响。')
edit(files[5],'from variables import InnerContext, InnerVariables','from variables import InnerContext, InnerVariables\nfrom solution_checks import CHECK_FIELDS, solution_checks')
edit(files[5],'    "pc3p_status", "gc3p_status",\n]','    "pc3p_status", "gc3p_status", "mosek_rel_gap",\n] + [prefix + key for prefix in ("pc3p_", "gc3p_") for key in CHECK_FIELDS]')
edit(files[5],'        "gc3p_status": gc3p_result["status"],','        "gc3p_status": gc3p_result["status"],\n        "mosek_rel_gap": params.mosek_tol_rel_gap,\n        **{"pc3p_" + k: v for k, v in solution_checks(pc3p_ctx, pc3p_result).items()},\n        **{"gc3p_" + k: v for k, v in solution_checks(gc3p_ctx, gc3p_result).items()},')
edit(files[5],'    return parser.parse_args()','    parser.add_argument("--mosek-rel-gap", type=float, default=None,\n                        help="PC3P/GC3P 统一的 MOSEK 相对间隙容差")\n    args = parser.parse_args()\n    if args.mosek_rel_gap is not None and not 0 < args.mosek_rel_gap <= 1:\n        parser.error("--mosek-rel-gap 必须在 (0, 1] 内")\n    return args')
edit(files[5],'    params = Parameters()\n    samples =','    params = Parameters()\n    if args.mosek_rel_gap is not None:\n        params.mosek_tol_rel_gap = args.mosek_rel_gap\n    print("MOSEK RelGap={:.1e}".format(params.mosek_tol_rel_gap))\n    samples =')
edit(files[5],'    # details 文件无需保存，此处直接注释掉（save_detail_csv 与 --detail-output 保留备用）\n    # if args.detail_output:\n    #     save_detail_csv(rows, args.detail_output)','    if args.detail_output:\n        save_detail_csv(rows, args.detail_output)')
print('Changed',files,'and solution_checks.py; backups:',B)
"""MOSEK Fusion 后端：与 cvxpy 后端共享同一套线性化量，用锥域等价改写求解 P5。

为什么要用锥域改写
------------------
本机 MOSEK Fusion（11.2.3 的 Python 绑定）只提供 **仿射** Expr：Expr 上没有
square/sqrt/exp/log/pow/inv 等非线性算子，Model.objective 也只接受仿射表达式（无二次目标）。
因此 cvxpy 中的 cp.square / cp.power(·,-2) / cp.inv_pos / cp.log 必须等价改写为原生锥约束：

    ũ_i ≥ f̃_i²                  → 旋转二阶锥 (ũ_i, 1/2, f̃_i) ∈ Q_r
    Q   ≥ Σ(z_i² + ũ_i²)        → 旋转二阶锥 (Q, 1/2, {z_i}, {ũ_i}) ∈ Q_r
    r_j ≥ 1/(D_j^max-D_j)       → 旋转二阶锥 (r_j, D_j^max-D_j, √2) ∈ Q_r
    s_j ≥ 1/(D_j^max-D_j)²      → 幂锥 α=1/3  (s_j, D_j^max-D_j, 1) ∈ P_{1/3}
    ℓ ≤ ln(Y)                   → 指数锥对偶 (Y, 1, ℓ) ∈ K_exp，即 x₁ ≥ x₂·e^{x₃/x₂}

复 Hermitian 变量：W_i = X_i + jY_i 拆成两个实矩阵（X 对称、Y 反对称），PSD 用实嵌入
[[X,-Y],[Y,X]] ⪰ 0。恒等式 Re Tr(A W) = <A_re, X> + <A_im, Y> 对应 Fusion 的
``Expr.dot(A_re, X) + Expr.dot(A_im, Y)``（二维 Expr.dot 为 Frobenius 内积）。

单一数据来源：所有随 CCCP 迭代变化的量仍由 ``cccp_params.compute_values`` 统一计算，
本模块只把它们写回 Fusion Parameter（``sync_fusion``），与 cvxpy 后端逐项同源。
"""

import numpy as np
import mosek.fusion as mf

from cccp_params import compute_values, cu_row_constants
from pc3p import (
    build_initial_point,
    compute_p4_objective,
    compute_rank1_gap,
    compute_rank1_gap_off,
    compute_rank1_gap_sen,
    recover_beamforming,
    update_linearization_points,
)

LN2 = np.log(2.0)
STRICT_TOL = 1e-10          # 与 constraints/c09 的 STRICT_TOL 保持一致


def _array(value):
    """转成 Fusion 可直接消费的 float64 连续数组。"""
    return np.ascontiguousarray(value, dtype=float)


def _real_trace(a_re, a_im, x, y):
    """Re Tr(A W)，W = X + jY：<A_re, X> + <A_im, Y>（a_re/a_im 可为 Parameter 或数组）。"""
    return mf.Expr.add(mf.Expr.dot(a_re, x), mf.Expr.dot(a_im, y))


class FusionCcpParams:
    """CCCP 线性化量对应的 Fusion Parameter 集合（对应 cccp_params.CccpParams）。"""

    def __init__(self, model, params):
        i_count, j_count, n = params.I, params.J, params.N
        self.d_prev_z_u = model.parameter("d_prev_z_u", i_count)
        self.cst_prev = model.parameter("cst_prev", i_count)
        self.bias_r1 = model.parameter("bias_r1", i_count)
        self.bias_r1_off = model.parameter("bias_r1_off", i_count)
        self.psi_sen_bias = model.parameter("psi_sen_bias", i_count)
        self.M_sen_re = [model.parameter("M_sen_re_%d" % i, [n, n]) for i in range(i_count)]
        self.M_sen_im = [model.parameter("M_sen_im_%d" % i, [n, n]) for i in range(i_count)]
        self.M_off_re = [model.parameter("M_off_re_%d" % i, [n, n]) for i in range(i_count)]
        self.M_off_im = [model.parameter("M_off_im_%d" % i, [n, n]) for i in range(i_count)]
        self.M_psi_re = [model.parameter("M_psi_re_%d" % i, [n, n]) for i in range(i_count)]
        self.M_psi_im = [model.parameter("M_psi_im_%d" % i, [n, n]) for i in range(i_count)]
        self.cu_r1 = model.parameter("cu_r1", j_count)
        self.cu_inv1 = model.parameter("cu_inv1", j_count)
        self.cu_r2 = model.parameter("cu_r2", j_count)
        self.cu_inv2 = model.parameter("cu_inv2", j_count)
        self.cu_k1 = model.parameter("cu_k1", j_count)
        self.cu_k2 = model.parameter("cu_k2", j_count)
        self.cu_p1 = model.parameter("cu_p1", j_count)
        self.cu_p2 = model.parameter("cu_p2", j_count)
        # ── 归一化参考量（全部为标量，逐次迭代刷新）────────────────────────────
        # MOSEK 内点法的可行性判据是相对量：Viol.con ≤ tol·max(1, nrm)，
        # 其中 nrm 是整个解向量的范数。本问题若不归一化，nrm ≈ Q ≈ 2e6，
        # 于是 tol=1e-9 在 c08 圆锥（量级 ~1e-4）上等价于允许 ~1e-2 的绝对违反
        # ⇒ 返回解违反 c08 达数百 bit。把这些"量级大但无物理意义"的副变量
        # 换成 O(1) 的归一化变量后 nrm ≈ O(10)，tol 恢复为有效的绝对容差。
        #   Q  = q_ref·Qn
        #   z  = z_ref·zn
        #   ũ  = u_ref·un
        self.q_ref = model.parameter("q_ref")
        self.inv_sqrt_q_ref = model.parameter("inv_sqrt_q_ref")
        self.z_ref = model.parameter("z_ref")
        self.u_ref = model.parameter("u_ref")
        self.inv_sqrt_u_ref = model.parameter("inv_sqrt_u_ref")
        self.zn_over_sqrt_qref = model.parameter("zn_over_sqrt_qref")
        self.un_over_sqrt_qref = model.parameter("un_over_sqrt_qref")
        # 最近一次 sync 使用的 z_ref / u_ref（供 run_pc3p_fusion 反归一化用）
        self.z_ref_now = 1.0
        self.u_ref_now = 1.0


class FusionVariables:
    """P5 的 Fusion 变量（复 Hermitian 矩阵拆成实部/虚部两个实矩阵）。"""

    def __init__(self, model, params):
        i_count, j_count, n = params.I, params.J, params.N
        self.X = [model.variable("X_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Y = [model.variable("Y_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Xb = [model.variable("Xb_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Yb = [model.variable("Yb_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.f_norm = model.variable("f_norm", i_count, mf.Domain.greaterThan(0.0))
        # ũ_i = u_ref·un_i、z_i = z_ref·zn_i（归一化变量，量级 O(1)；见 FusionCcpParams）
        self.un = model.variable("un", i_count, mf.Domain.greaterThan(0.0))
        self.zn = model.variable("zn", i_count, mf.Domain.greaterThan(0.0))
        self.D = model.variable("D", j_count, mf.Domain.greaterThan(0.0))
        # 目标 ①② 的锥辅助变量（Fusion 目标必须仿射）
        # Qn 为归一化能耗：Q = q_ref·Qn，使变量量级 O(1)（q_ref 见 FusionCcpParams）
        self.Qn = model.variable("Qn", 1, mf.Domain.greaterThan(0.0))
        self.s = model.variable("s", j_count, mf.Domain.greaterThan(0.0))
        self.r = model.variable("r", j_count, mf.Domain.greaterThan(0.0))
        # c08 / c12 中 log 的下界辅助变量：(·, 1, ℓ) ∈ K_exp ⇔ ℓ ≤ ln(·)
        self.ell_sen = model.variable("ell_sen", i_count, mf.Domain.unbounded())
        self.ell_w = model.variable("ell_w", j_count, mf.Domain.unbounded())
        self.ell_b = model.variable("ell_b", j_count, mf.Domain.unbounded())


def sync_fusion(model, fp, ctx):
    """由当前线性化点刷新全部 Fusion Parameter 取值（数据来源与 cvxpy 后端相同）。"""
    v = compute_values(ctx)
    fp.d_prev_z_u.setValue(_array(v["d_prev_z_u"]))
    fp.cst_prev.setValue(_array(v["cst_prev"]))
    fp.bias_r1.setValue(_array(v["bias_r1"]))
    fp.bias_r1_off.setValue(_array(v["bias_r1_off"]))
    fp.psi_sen_bias.setValue(_array(v["psi_sen_bias"]))
    for i in range(ctx.I):
        fp.M_sen_re[i].setValue(_array(v["M_sen_re"][i]))
        fp.M_sen_im[i].setValue(_array(v["M_sen_im"][i]))
        fp.M_off_re[i].setValue(_array(v["M_off_re"][i]))
        fp.M_off_im[i].setValue(_array(v["M_off_im"][i]))
        fp.M_psi_re[i].setValue(_array(v["M_psi_re"][i]))
        fp.M_psi_im[i].setValue(_array(v["M_psi_im"][i]))
    for name in ("cu_r1", "cu_r2", "cu_inv1", "cu_inv2", "cu_k1", "cu_k2", "cu_p1", "cu_p2"):
        getattr(fp, name).setValue(_array(v[name]))
    # 归一化参考量：取当前线性化点的量级作为参考。这些参考量同时出现在
    # 约束与目标中，逐项恒等相消，故取值不改变最优解/可行域，只决定归一化
    # 变量（Qn/zn/un）的数值量级——这正是压低 nrm 的关键。
    z_prev = np.asarray(ctx.z_aux_rate_prev, dtype=float).reshape(-1)
    u_prev = (np.asarray(ctx.f_uav_freq_prev, dtype=float).reshape(-1) / ctx.freq_scale) ** 2
    z_ref = max(float(np.max(np.abs(z_prev))) if z_prev.size else 0.0, 1.0)
    u_ref = max(float(np.max(np.abs(u_prev))) if u_prev.size else 0.0, 1e-6)
    q_ref = max(float(np.sum(z_prev ** 2 + u_prev ** 2)), 1.0)
    fp.z_ref.setValue(z_ref)
    fp.u_ref.setValue(u_ref)
    fp.inv_sqrt_u_ref.setValue(1.0 / np.sqrt(u_ref))
    fp.q_ref.setValue(q_ref)
    fp.inv_sqrt_q_ref.setValue(1.0 / np.sqrt(q_ref))
    fp.zn_over_sqrt_qref.setValue(z_ref / np.sqrt(q_ref))
    fp.un_over_sqrt_qref.setValue(u_ref / np.sqrt(q_ref))
    fp.z_ref_now = z_ref
    fp.u_ref_now = u_ref
    model.flushParameters()


def _sum_terms(terms):
    """合并若干 Fusion 表达式/常数。"""
    acc = None
    for term in terms:
        if isinstance(term, (int, float)):
            if term == 0.0:
                continue
            term = mf.Expr.constTerm(float(term))
        acc = term if acc is None else mf.Expr.add(acc, term)
    return acc


def build_objective_fusion(ctx, fp, fv):
    """构造 P5 目标：与 objective.build_objective 逐项对应，非线性项用锥辅助变量替代。"""
    omega_1, omega_2, omega_3 = ctx.omega_1, ctx.omega_2, ctx.omega_3
    g_sen = omega_1 * ctx.kappa_cpu * ctx.C_sen * ctx.freq_scale ** 2
    trace_id = np.eye(ctx.N)

    terms = []
    # ① 感知计算能耗 (g_sen/2)·Q，Q ≥ Σ(z_i² + ũ_i²)（锥约束见 build_constraints_fusion）
    #   等价的归一化写法：Q = q_ref·Qn ⇒ 该常数 q_ref 由 Parameter 提供（见 sync_fusion）
    terms.append(mf.Expr.mul(g_sen / 2.0, mf.Expr.mul(fp.q_ref, fv.Qn.index(0))))

    # ⑤ 乘积 g·z_i·ũ_i 的 CCCP 线性化：- g·Σ d_i(z_i - ũ_i) + g·Σ cst_i
    #   代入 z = z_ref·zn、ũ = u_ref·un（恒等）
    terms.append(mf.Expr.mul(-g_sen, mf.Expr.mul(fp.z_ref, mf.Expr.dot(fp.d_prev_z_u, fv.zn))))
    terms.append(mf.Expr.mul(g_sen, mf.Expr.mul(fp.u_ref, mf.Expr.dot(fp.d_prev_z_u, fv.un))))
    terms.append(mf.Expr.mul(g_sen, mf.Expr.sum(fp.cst_prev)))

    # ② 娱乐计算能耗 Σ ω₁κ(C_jL_j)³·s_j，s_j ≥ 1/(D_j^max-D_j)²（幂锥给出）
    coeff2 = omega_1 * ctx.kappa_cpu * (ctx.C_cu_cycles * ctx.L_cu_task) ** 3
    terms.append(mf.Expr.dot(_array(coeff2), fv.s))

    # ③ UAV 感知/卸载/飞行能耗
    for i in range(ctx.I):
        per_uav = _sum_terms([
            mf.Expr.mul(ctx.D_bar_sen, mf.Expr.dot(trace_id, fv.X[i])),
            mf.Expr.mul(ctx.D_uav_off[i], mf.Expr.dot(trace_id, fv.Xb[i])),
            ctx.E_uav_fly[i],
        ])
        terms.append(mf.Expr.mul(omega_2, per_uav))

    # ④ CU 卸载能耗
    terms.append(mf.Expr.mul(omega_3, mf.Expr.dot(_array(ctx.p_cu_power), fv.D)))

    # ⑥ 秩一罚项 Re Tr(M_i W_i) + bias_i
    for i in range(ctx.I):
        terms.append(mf.Expr.add(
            _real_trace(fp.M_sen_re[i], fp.M_sen_im[i], fv.X[i], fv.Y[i]), fp.bias_r1.index(i)))
        terms.append(mf.Expr.add(
            _real_trace(fp.M_off_re[i], fp.M_off_im[i], fv.Xb[i], fv.Yb[i]), fp.bias_r1_off.index(i)))

    return _sum_terms(terms)


# 对称 / 反对称方程的独立索引（行, 列）缓存，键为 N（避免每次建模重复构造）：
#   strict_upper    —— 严格上三角 (i<j)，共 N(N-1)/2 个，用于 X 对称；
#   upper_incl_diag —— 上三角含对角 (i≤j)，共 N(N+1)/2 个，用于 Y 反对称。
_HERMITIAN_INDEX_CACHE = {}


def _hermitian_index_rows(n):
    """返回 (strict_upper, upper_incl_diag) 两组供 ``Expr.pick`` 使用的 (行, 列) 索引数组。"""
    cached = _HERMITIAN_INDEX_CACHE.get(n)
    if cached is None:
        strict_upper = np.array([(a, b) for a in range(n) for b in range(a + 1, n)],
                                dtype=np.int32).reshape(-1, 2)
        upper_incl_diag = np.array([(a, b) for a in range(n) for b in range(a, n)],
                                   dtype=np.int32).reshape(-1, 2)
        cached = (strict_upper, upper_incl_diag)
        _HERMITIAN_INDEX_CACHE[n] = cached
    return cached


def _add_hermitian_psd(model, x, y, tag, n):
    """X 对称、Y 反对称，PSD 用实嵌入 [[X,-Y],[Y,X]] ⪰ 0 表达 W = X + jY ⪰ 0。

    对称 / 反对称只写**独立**方程，而不是整块 N×N 矩阵等式：

        X - Xᵀ 在严格上三角 (i<j) 上为零   —— X 对称，共 N(N-1)/2 个独立方程；
        Y + Yᵀ 在上三角含对角 (i≤j) 上为零 —— Y 反对称，共 N(N+1)/2 个独立方程
                                              （对角线上是 Y_ii + Y_ii = 0，同样强制 Y 对角为零）。

    原来写成整块 N×N 等式时，每个矩阵要写 2·N² 行，其中 i>j 的行只是 i<j 行的重复、
    i=j 行的残差恒为零；改写成独立方程后每个矩阵只需 N² 行。本问题有 8 个 Hermitian
    矩阵（4 个 W_i + 4 个 B_i），优化器问题里的线性约束行数因此从 1600 降到 800
    （optNumcon 实测 1649 → 849）。两者可行域完全相同（只是把重复 / 恒零的行去掉），
    故最优值不变；实测每次内点迭代的线性代数开销下降约 9%，解向量与旧写法逐位一致。
    """
    # 注：下面两处 Expr.pick 会被类型检查器报 reportArgumentType —— MOSEK 的类型注解把
    # Expression.pick 的接收者声明为 Expr，而 Expr.sub / Expr.add 的返回类型是
    # ExprWSum | Expr | ExprAdd 联合，属静态误报（运行时行为已实测验证）。故就地抑制。
    strict_upper, upper_incl_diag = _hermitian_index_rows(n)
    if strict_upper.size:
        # X - Xᵀ 的严格上三角为零 ⇔ X 对称（下三角是对称的重复行，对角恒为零）
        model.constraint("sym_%s" % tag,
                         mf.Expr.pick(mf.Expr.sub(x, x.transpose()),  # pyright: ignore[reportArgumentType]
                                      strict_upper),
                         mf.Domain.equalsTo(0.0))
    # Y + Yᵀ 的上三角（含对角）为零 ⇔ Y 反对称（对角上是 2Y_ii = 0）
    model.constraint("asym_%s" % tag,
                     mf.Expr.pick(mf.Expr.add(y, y.transpose()),  # pyright: ignore[reportArgumentType]
                                  upper_incl_diag),
                     mf.Domain.equalsTo(0.0))
    top = mf.Expr.hstack(x, mf.Expr.neg(y))
    bottom = mf.Expr.hstack(y, x)
    model.constraint("psd_%s" % tag, mf.Expr.vstack(top, bottom), mf.Domain.inPSDCone(2 * n))


def build_constraints_fusion(ctx, model, fp, fv):
    """构造全部约束，逐条对应 constraints/c01..c13。"""
    freq_scale = ctx.freq_scale
    trace_id = np.eye(ctx.N)
    f_uav = [mf.Expr.mul(freq_scale, fv.f_norm.index(i)) for i in range(ctx.I)]

    # ── c01 Task-Fresh：D̄^sen + D_off - Σ_j η_ij D_j ≤ 0 ────────────────────
    for i in range(ctx.I):
        expr = mf.Expr.sub(ctx.D_bar_sen + ctx.D_uav_off[i],
                           mf.Expr.dot(_array(ctx.eta_share[i, :]), fv.D))
        model.constraint("c01_task_fresh_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c02 UAV-task-delay：(D̄^sen + D_off - D_sen^max)·f_i + C_sen·z_i ≤ 0 ──
    for i in range(ctx.I):
        coeff = ctx.D_bar_sen + ctx.D_uav_off[i] - ctx.D_max_sen
        expr = mf.Expr.add(mf.Expr.mul(coeff, f_uav[i]),
                           mf.Expr.mul(ctx.C_sen, mf.Expr.mul(fp.z_ref, fv.zn.index(i))))
        model.constraint("c02_uav_task_delay_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c03 / c04 半正定 ─────────────────────────────────────────────────────
    for i in range(ctx.I):
        _add_hermitian_psd(model, fv.X[i], fv.Y[i], "sen_%d" % i, ctx.N)
        _add_hermitian_psd(model, fv.Xb[i], fv.Yb[i], "off_%d" % i, ctx.N)

    # ── c05 UAV-Sen-Max-Power：Tr(W_i) ≤ P_max ──────────────────────────────
    for i in range(ctx.I):
        model.constraint("c05_sen_pow_%d" % i,
                         mf.Expr.sub(mf.Expr.dot(trace_id, fv.X[i]), ctx.P_max_uav),
                         mf.Domain.lessThan(0.0))

    # ── c06 Sensing-SINR：εΓ_i - Re Tr(G_i W_i) ≤ 0 ─────────────────────────
    for i in range(ctx.I):
        g_mat = ctx.G_sen_corr[i]
        expr = mf.Expr.sub(ctx.eps_sinr * ctx.Gamma_sinr[i],
                           _real_trace(_array(g_mat.real), _array(g_mat.imag),
                                       fv.X[i], fv.Y[i]))
        model.constraint("c06_sensing_sinr_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c07 UAV-Off-Max-Power：Tr(B_i) ≤ P_max ──────────────────────────────
    for i in range(ctx.I):
        model.constraint("c07_off_pow_%d" % i,
                         mf.Expr.sub(mf.Expr.dot(trace_id, fv.Xb[i]), ctx.P_max_uav),
                         mf.Domain.lessThan(0.0))

    # ── c08 UAVs-task-successful-offloading ─────────────────────────────────
    #     原式 z_i - D_off·B·log_2(Tr(H B_i)+Φ_i) + D_off·B·log_2(Φ_i) ≤ 0。
    #     因 Φ_i ~ 1e-12、D_off·B ~ 1e5，原式的 log_2(Φ_i) 常数高达 ~1e7，
    #     与同量级的 log 项相消到 O(10²) 才得到 z_i，求解器在 1e7 量级上以
    #     相对容差 1e-7 工作 ⇒ 有效误差 ~1，直接病态（IllPosed / PrimalInfeasible）。
    #     这里做与 c12 完全相同的恒等归一化：把 "1 + ·" 移入指数锥（log 自变量 ~1），
    #     再把整行除以 D_off·B/ln2：
    #         z_i·ln2/(D_off·B) - ℓ_i ≤ 0,   ℓ_i ≤ ln(1 + Tr(H B_i)/Φ_i)
    #     与原式逐项恒等，但所有系数回到 O(1)。
    for i in range(ctx.I):
        h_mat = ctx.H_uav_bs[i]
        phi = ctx.Phi_off_inr[i]
        row_scale = ctx.D_uav_off[i] * ctx.B / LN2     # 原式除以它做逐行归一化
        arg_y = mf.Expr.add(
            1.0,
            mf.Expr.mul(1.0 / phi,
                        _real_trace(_array(h_mat.real), _array(h_mat.imag),
                                    fv.Xb[i], fv.Yb[i])))
        model.constraint("c08_exp_%d" % i,
                         mf.Expr.vstack(arg_y, 1.0, fv.ell_sen.index(i)),
                         mf.Domain.inPExpCone())
        model.constraint("c08_%d" % i,
                         mf.Expr.sub(mf.Expr.mul(1.0 / row_scale, mf.Expr.mul(fp.z_ref, fv.zn.index(i))),
                                     fv.ell_sen.index(i)),
                         mf.Domain.lessThan(0.0))

    # ── c09 CU-Frequency-Always-Positive：D_j - D_j^max + STRICT_TOL ≤ 0 ────
    for j in range(ctx.J):
        model.constraint("c09_%d" % j,
                         mf.Expr.sub(fv.D.index(j), ctx.D_max_cu[j] - STRICT_TOL),
                         mf.Domain.lessThan(0.0))

    # ── c10 BS-Max-Frequency ────────────────────────────────────────────────
    #     r_j ≥ 1/(D_j^max - D_j)（旋转二阶锥），(Σ f + Σ C L r - F_max)/freq_scale ≤ 0
    d_slack = [mf.Expr.sub(ctx.D_max_cu[j], fv.D.index(j)) for j in range(ctx.J)]
    for j in range(ctx.J):
        model.constraint("c10_rcone_%d" % j,
                         mf.Expr.vstack(fv.r.index(j), d_slack[j], np.sqrt(2.0)),
                         mf.Domain.inRotatedQCone())
    freq_terms = list(f_uav)
    for j in range(ctx.J):
        freq_terms.append(mf.Expr.mul(ctx.C_cu_cycles[j] * ctx.L_cu_task[j], fv.r.index(j)))
    model.constraint("c10_bs_freq",
                     mf.Expr.mul(1.0 / freq_scale,
                                 mf.Expr.sub(_sum_terms(freq_terms), ctx.F_max)),
                     mf.Domain.lessThan(0.0))

    # ── c11 CCCP:Auxiliary-Variable-1：-z_i + Re Tr(M_psi W_i) + bias ≤ 0 ──
    for i in range(ctx.I):
        expr = mf.Expr.add(
            mf.Expr.neg(mf.Expr.mul(fp.z_ref, fv.zn.index(i))),
            _real_trace(fp.M_psi_re[i], fp.M_psi_im[i], fv.X[i], fv.Y[i]))
        expr = mf.Expr.add(expr, fp.psi_sen_bias.index(i))
        model.constraint("c11_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c12 CCCP:CUs-task-successful-offloading ─────────────────────────────
    w1, w2, dw, const, _ = cu_row_constants(ctx)
    for j in range(ctx.J):
        eta_col = ctx.eta_share[:, j]
        terms_w, terms_b = [], []
        for i in range(ctx.I):
            if eta_col[i] == 0.0:
                continue
            h_mat = ctx.H_uav_bs[i]
            terms_w.append(mf.Expr.mul(eta_col[i], _real_trace(
                _array(h_mat.real), _array(h_mat.imag), fv.X[i], fv.Y[i])))
            terms_b.append(mf.Expr.mul(eta_col[i], _real_trace(
                _array(h_mat.real), _array(h_mat.imag), fv.Xb[i], fv.Yb[i])))

        expr = mf.Expr.add(mf.Expr.mul(-dw[j], fv.D.index(j)), const[j])
        if w1[j] > 0.0:
            sum_w = _sum_terms(terms_w)
            arg_w = mf.Expr.add(fp.cu_r1.index(j), mf.Expr.mul(fp.cu_inv1.index(j), sum_w))
            model.constraint("c12_exp_w_%d" % j,
                             mf.Expr.vstack(arg_w, 1.0, fv.ell_w.index(j)),
                             mf.Domain.inPExpCone())
            expr = mf.Expr.add(expr, mf.Expr.mul(-w1[j] / LN2, fv.ell_w.index(j)))
            expr = mf.Expr.add(expr, mf.Expr.mul(fp.cu_k1.index(j), sum_w))
            expr = mf.Expr.sub(expr, fp.cu_p1.index(j))
        if w2[j] > 0.0:
            sum_b = _sum_terms(terms_b)
            arg_b = mf.Expr.add(fp.cu_r2.index(j), mf.Expr.mul(fp.cu_inv2.index(j), sum_b))
            model.constraint("c12_exp_b_%d" % j,
                             mf.Expr.vstack(arg_b, 1.0, fv.ell_b.index(j)),
                             mf.Domain.inPExpCone())
            expr = mf.Expr.add(expr, mf.Expr.mul(-w2[j] / LN2, fv.ell_b.index(j)))
            expr = mf.Expr.add(expr, mf.Expr.mul(fp.cu_k2.index(j), sum_b))
            expr = mf.Expr.sub(expr, fp.cu_p2.index(j))
        model.constraint("c12_%d" % j, expr, mf.Domain.lessThan(0.0))

    # ── c13 Epigraph-Tau：square(f_i/freq_scale) - ũ_i ≤ 0，即 f̃_i² ≤ ũ_i ──
    for i in range(ctx.I):
        model.constraint("c13_%d" % i,
                         mf.Expr.vstack(fv.un.index(i), 0.5,
                                        mf.Expr.mul(fp.inv_sqrt_u_ref, fv.f_norm.index(i))),
                         mf.Domain.inRotatedQCone())

    # ── 锥辅助：① Q ≥ Σ(z_i² + ũ_i²)、② s_j ≥ 1/(D_j^max-D_j)² ─────────────
    #     Fusion 的 vstack 最多接受 3 个参数，长向量用两两折叠拼接
    #     归一化写法（与 Q ≥ Σ(z²+ũ²) 恒等）：(Qn, 1/2, z/√q_ref, ũ/√q_ref) ∈ Q_r
    #     ⇔ Qn ≥ Σ(z_i²+ũ_i²)/q_ref ⇔ q_ref·Qn ≥ Σ(z_i²+ũ_i²)。
    #     把 √q_ref 折进 c 分量，避免 Q 变量本身取到 ~1e6 而抬高 nrm。
    q_vec = mf.Expr.vstack(fv.Qn.index(0), 0.5)
    q_vec = mf.Expr.vstack(q_vec, mf.Expr.mul(fp.zn_over_sqrt_qref, fv.zn))
    q_vec = mf.Expr.vstack(q_vec, mf.Expr.mul(fp.un_over_sqrt_qref, fv.un))
    model.constraint("obj_q_cone", q_vec, mf.Domain.inRotatedQCone())
    #     s_j ≥ 1/d_j² 可等价改写为 s_j ≥ r_j²：因 c10 已令 r_j ≥ 1/d_j，
    #     故 s_j ≥ r_j² ⟺ s_j ≥ 1/d_j²（r_j 会在最优处取到 1/d_j）。
    #     用旋转二阶锥替代幂锥 α=1/3，避免幂锥在最优点附近的数值退化。
    for j in range(ctx.J):
        model.constraint("obj_s_cone_%d" % j,
                         mf.Expr.vstack(fv.s.index(j), 0.5, fv.r.index(j)),
                         mf.Domain.inRotatedQCone())


def build_fusion_model(ctx):
    """构建并返回 (model, FusionCcpParams, FusionVariables)，模型只建一次、后续复用。"""
    model = mf.Model("P5_fusion")
    fp = FusionCcpParams(model, ctx.params)
    fv = FusionVariables(model, ctx.params)
    build_constraints_fusion(ctx, model, fp, fv)
    model.objective(mf.ObjectiveSense.Minimize, build_objective_fusion(ctx, fp, fv))
    return model, fp, fv


def _hermitian_from(x_level, y_level, n):
    """由实部/虚部变量取值还原 Hermitian 矩阵：W = X + jY，再强制 Hermitian。"""
    x = np.asarray(x_level, dtype=float).reshape(n, n)
    y = np.asarray(y_level, dtype=float).reshape(n, n)
    w = x + 1j * y
    return (w + w.conj().T) / 2.0


def run_pc3p_fusion(ctx):
    """MOSEK Fusion 版 PC3P：迭代结构与 run_pc3p 完全一致，仅求解后端不同。"""
    # ρ 复位：ctx.params 可能被 main 的循环复用，必须回到初值，否则会跨样本累积
    ctx.params.rho_penalty = ctx.rho_penalty_init
    build_initial_point(ctx)
    x_prev = compute_p4_objective(ctx, ctx.W_sen_beam_prev, ctx.B_off_beam_prev,
                                  ctx.f_uav_freq_prev, ctx.z_aux_rate_prev, ctx.D_cu_off_prev)

    result = {
        "status": None,
        "iterations": 0,
        "converged": False,
        "objective_p4": x_prev,
        "rank1_gap": None,
        "W_sen_beam": None,
        "B_off_beam": None,
        "w_sen_beam": None,
        "b_off_beam": None,
        "f_uav_freq": None,
        "f_cu_freq": None,
        "z_aux_rate": None,
        "D_cu_off": None,
        "rho_final": ctx.rho_penalty,
        "obj_history": [x_prev],
        "w_gap_history": [compute_rank1_gap_sen(ctx.W_sen_beam_prev)],
        "b_gap_history": [compute_rank1_gap_off(ctx.B_off_beam_prev)],
    }

    model, fp, fv = build_fusion_model(ctx)
    # 先按 x^(0) 写参数，之后每轮只 flushParameters 复用同一模型
    update_linearization_points(ctx)
    sync_fusion(model, fp, ctx)
    # Fusion 的求解器参数名采用 camelCase（无 MSK_ 前缀），
    # 与 cvxpy 侧 mosek_params 里的 MSK_DPAR_INTPNT_CO_TOL_PFEAS/DFEAS 等价
    model.setSolverParam("intpntCoTolPfeas", ctx.mosek_tol_feas)
    model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)

    for n in range(1, ctx.max_iterations + 1):
        try:
            model.solve()
        except Exception as exc:                                  # noqa: BLE001
            result["status"] = "error: %s" % exc
            result["iterations"] = n
            return result

        status = model.getProblemStatus()
        result["status"] = str(status)
        result["iterations"] = n
        # 与 run_pc3p 的控制流一致：非最优即返回
        if status != mf.ProblemStatus.PrimalAndDualFeasible:
            return result

        # x^(n) ← x^⋆
        W_val = np.array([_hermitian_from(fv.X[i].level(), fv.Y[i].level(), ctx.N)
                          for i in range(ctx.I)])
        B_val = np.array([_hermitian_from(fv.Xb[i].level(), fv.Yb[i].level(), ctx.N)
                          for i in range(ctx.I)])
        f_val = ctx.freq_scale * np.asarray(fv.f_norm.level(), dtype=float).reshape(ctx.I)
        z_val = fp.z_ref_now * np.asarray(fv.zn.level(), dtype=float).reshape(ctx.I)
        D_val = np.asarray(fv.D.level(), dtype=float).reshape(ctx.J)

        ctx.W_sen_beam_prev[:] = W_val
        ctx.B_off_beam_prev[:] = B_val
        ctx.f_uav_freq_prev[:] = f_val
        ctx.z_aux_rate_prev[:] = z_val
        ctx.D_cu_off_prev[:] = D_val

        x_n = compute_p4_objective(ctx, W_val, B_val, f_val, z_val, D_val)
        gap_n = compute_rank1_gap(W_val, B_val)
        result["objective_p4"] = x_n
        result["rank1_gap"] = gap_n
        result["obj_history"].append(x_n)
        result["w_gap_history"].append(compute_rank1_gap_sen(W_val))
        result["b_gap_history"].append(compute_rank1_gap_off(B_val))

        if abs(x_n - x_prev) <= ctx.gamma_1 * abs(x_prev) and gap_n <= ctx.gamma_2:
            result["converged"] = True
            x_prev = x_n
            break
        x_prev = x_n

        if gap_n > ctx.gamma_2 and ctx.rho_penalty < ctx.rho_penalty_max:
            new_rho = min(ctx.rho_penalty * ctx.rho_penalty_scale, ctx.rho_penalty_max)
            if new_rho > ctx.rho_penalty:
                ctx.params.rho_penalty = new_rho
                result["rho_final"] = new_rho

        update_linearization_points(ctx)
        sync_fusion(model, fp, ctx)

    w_sen_beam, b_off_beam = recover_beamforming(ctx.W_sen_beam_prev, ctx.B_off_beam_prev)
    f_cu_freq = ctx.C_cu_cycles * ctx.L_cu_task / (ctx.D_max_cu - ctx.D_cu_off_prev)

    result["W_sen_beam"] = ctx.W_sen_beam_prev
    result["B_off_beam"] = ctx.B_off_beam_prev
    result["w_sen_beam"] = w_sen_beam
    result["b_off_beam"] = b_off_beam
    result["f_uav_freq"] = ctx.f_uav_freq_prev
    result["f_cu_freq"] = f_cu_freq
    result["z_aux_rate"] = ctx.z_aux_rate_prev
    result["D_cu_off"] = ctx.D_cu_off_prev
    return result

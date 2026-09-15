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

模型只建一次（进程内复用）
--------------------------
P5 的**结构**只由 (I, J, N) 决定；把它作为缓存键，整个进程只创建一次 ``mf.Model``
（= 一个 MOSEK Task），之后每步 / 每轮只需 ``setValue`` + ``flushParameters``。
之所以必须这样做（实测问题）：

    ``mf.Model`` 内部持有原生 MOSEK Task、表达式池（``_Model__xs/__ws/__rs``）与
    解数组（``_sol_itr`` 等），且 Model 与 Variable/Parameter/Constraint 之间存在
    引用环（``BaseVariable.__slots__`` 含 ``_BaseVariable__model``，Model 又持有
    ``__vars/__cons/__xs``），因此这些原生内存只能等循环 GC 才回收。外层
    ``my_env.step`` **每个时隙**都会走到 ``build_fusion_model``，长训练
    （8000 episode × 40 时隙）下原生内存单调增长，最终 numpy 连 1.6 MiB 都分配不出来，
    抛 ``numpy.core._exceptions._ArrayMemoryError``。复用单一模型后内存为 O(1)。

参数化约定：逐叶替换，保持表达式树不变
--------------------------------------
为了让复用模型与"每步重建"的旧实现在**数值上逐位一致**，本模块采用"逐叶替换"策略：

    - 旧实现里每个来自 ctx 的 numpy 常数（标量或矩阵），在这里都换成取值完全相同的
      ``model.parameter``；**表达式树、乘法顺序、相加顺序都与旧实现逐字对应**。
    - 因此不会出现"参数 × 参数"的非法乘积：旧实现里"两个随场景变化的量相乘"的位置，
      本来就是"常数 × 参数"或"常数 × 变量"，换成"参数 × 参数 / 参数 × 变量"即可，
      两者都是 Fusion 允许的仿射标量缩放。
    - 不在 numpy 侧预先折叠（例如不把 ``omega_2 * D_uav_off`` 先算成一个数），
      否则会改变浮点舍入顺序，使结果出现 ~1e-16 的系数扰动；在本问题这种近乎退化的
      最优解集上，这种扰动会被放大成可见的解分量差异。

唯一的**结构性**改动在 c12：旧实现按当前时隙的 η 匹配模式决定是否建项
（``if eta_col[i] == 0: continue`` / ``if w1[j] > 0`` / ``if w2[j] > 0``），
结构随时隙变化 ⇒ 无法复用。这里改为**总是**建全部项与两个指数锥，用参数（η→0、
w1/w2→0）把这些项系数置零：数值上等价（0 系数贡献 0），但固定了模型结构。

改造前后的一致性（实测，验证脚本已删除，结论记录在此）
------------------------------------------------------
对比「git HEAD 的旧实现」与「本实现」，用 16 组外层样本（12 组代回 P1 全约束通过 +
4 组随机）在两套实现上求解：

    1) 模型规模（底层 MOSEK Task）：两侧均为 849 行 / 1668 列；
    2) 模型数据：把两侧任务导出为 PTF 后逐系数比对（按变量汇集、组内排序以消除
       "项的书写顺序"影响）——真实 η 场景 7762 条系数、η 全正值场景 35121 条系数，
       **最大相对差 0.000e+00**（差异仅为 MOSEK 对常数项的等价写法：``+0.6`` 与
       ``+0.6 '1.0'``）；
    3) 求解结果：状态、迭代次数、是否收敛、罚因子 ρ 全部一致；P4 目标值（即奖励所用
       的能量口径）相对差 ≤ 3e-8（内点法容差量级）。把 c12 结构对齐（令所有 η>0，
       使旧实现的分支恰好全部命中）后，连解向量与迭代历史也**逐位一致**（相对差 0.0）；
    4) 生产路径（``outer/reward/cccp_bridge.solve_inner_energy``）：energy 相对差
       ≤ 3.5e-9，返回的各明细列表在求解器容差内一致。

残留差异的来源是 c12 的"超集结构"（旧实现逐时隙增删项）改变了 MOSEK 内部的
稀疏序与 `@ac` 编号，使内点法路径不同；这只会体现在退化方向上的分量（如 f_uav、
z_aux_rate、接近 0 的卸载功率）上，不影响最优值与能量口径。

内存（实测 120 次求解，Windows / psutil 读 RSS）：
    旧实现（每时隙新建、从不 dispose）  +7.000 MB/次求解（97 MB → 961 MB，线性增长）；
    本实现（单模型复用）               +0.028 MB/次求解（一次性 +42 MB 后稳定在 139 MB）。
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
SQRT2 = np.sqrt(2.0)
STRICT_TOL = 1e-10          # 与 constraints/c09 的 STRICT_TOL 保持一致


def _array(value):
    """转成 Fusion 可直接消费的 float64 连续数组。"""
    return np.ascontiguousarray(value, dtype=float)


def _real_trace(a_re, a_im, x, y):
    """Re Tr(A W)，W = X + jY：<A_re, X> + <A_im, Y>（a_re/a_im 可为 Parameter 或数组）。"""
    return mf.Expr.add(mf.Expr.dot(a_re, x), mf.Expr.dot(a_im, y))


class FusionParams:
    """P5 模型的全部 Parameter，与旧实现的每个 numpy 常数一一对应。

    命名尽量沿用旧实现里的局部变量名（g_sen / coeff2 / row_scale …），便于逐条对照。
    分四类（都在 ``sync_fusion`` 中刷新）：
      (a) CCCP 线性化量   —— 每轮 CCCP 迭代刷新（来源 ``cccp_params.compute_values``）；
      (b) 归一化参考量     —— 每轮刷新（z_ref / u_ref / q_ref 及其派生量）；
      (c) 目标函数系数     —— 每时隙刷新；
      (d) 约束系数         —— 每时隙刷新。
    """

    def __init__(self, model, params):
        i_count, j_count, n = params.I, params.J, params.N
        self.I, self.J, self.N = i_count, j_count, n

        # ── (a) CCCP 线性化量（对应 cccp_params.CccpParams）──────────────────
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

        # ── (b) 归一化参考量（全部为标量，逐次迭代刷新）──────────────────────
        # MOSEK 内点法的可行性判据是相对量：Viol.con ≤ tol·max(1, nrm)，
        # 其中 nrm 是整个解向量的范数。本问题若不归一化，nrm ≈ Q ≈ 2e6，
        # 于是 tol=1e-9 在 c08 圆锥（量级 ~1e-4）上等价于允许 ~1e-2 的绝对违反
        # ⇒ 返回解违反 c08 达数百 bit。把这些"量级大但无物理意义"的副变量
        # 换成 O(1) 的归一化变量后 nrm ≈ O(10)，tol 恢复为有效的绝对容差。
        #   Q  = q_ref·Qn
        #   z  = z_ref·zn
        #   ũ  = u_ref·un
        self.q_ref = model.parameter("q_ref")
        self.z_ref = model.parameter("z_ref")
        self.u_ref = model.parameter("u_ref")
        self.inv_sqrt_u_ref = model.parameter("inv_sqrt_u_ref")
        self.zn_over_sqrt_qref = model.parameter("zn_over_sqrt_qref")
        self.un_over_sqrt_qref = model.parameter("un_over_sqrt_qref")
        # 最近一次 sync 使用的 z_ref（供 run_pc3p_fusion 反归一化 z 用）
        self.z_ref_now = 1.0

        # ── (c) 目标函数系数（每时隙刷新）────────────────────────────────────
        # ① / ⑤：旧实现里的 g_sen = ω₁κC^sen f_scale²（标量），三个用法各存一个取值
        self.obj_g_sen_half = model.parameter("obj_g_sen_half")      # g_sen / 2
        self.obj_g_sen = model.parameter("obj_g_sen")                # g_sen
        self.obj_neg_g_sen = model.parameter("obj_neg_g_sen")        # -g_sen
        # ②：coeff2 = ω₁κ(C_jL_j)³
        self.obj_coeff2 = model.parameter("obj_coeff2", j_count)
        # ③：ω₂ · [ D̄^sen Tr(W_i) + D_i^off Tr(B_i) + E_i^fly ]
        self.obj_omega_2 = model.parameter("obj_omega_2")
        self.obj_dbar = model.parameter("obj_dbar")
        self.obj_doff = model.parameter("obj_doff", i_count)
        self.obj_efly = model.parameter("obj_efly", i_count)
        # ④：ω₃ · Σ_j p_j D_j
        self.obj_omega_3 = model.parameter("obj_omega_3")
        self.obj_p_cu = model.parameter("obj_p_cu", j_count)

        # ── (d) 约束系数（每时隙刷新）────────────────────────────────────────
        # c01：D̄^sen + D_i^off - Σ_j η_ij D_j ≤ 0
        self.c01_rhs = model.parameter("c01_rhs", i_count)
        self.c01_eta = [model.parameter("c01_eta_%d" % i, j_count) for i in range(i_count)]
        # c02：(D̄^sen + D_i^off - D^sen_max)·f_i + C^sen·z_i ≤ 0，f_i = f_scale·f̃_i
        self.c02_coeff = model.parameter("c02_coeff", i_count)
        self.c02_c_sen = model.parameter("c02_c_sen")
        self.freq_scale = model.parameter("freq_scale")
        # c05 / c07：Tr(·) ≤ P^max_UAV（旧实现两处用同一常数，这里共用一个参数）
        self.p_max_uav = model.parameter("p_max_uav")
        # c06：ε·Γ_i - Re Tr(G_i W_i) ≤ 0
        self.c06_rhs = model.parameter("c06_rhs", i_count)
        self.G_re = [model.parameter("G_re_%d" % i, [n, n]) for i in range(i_count)]
        self.G_im = [model.parameter("G_im_%d" % i, [n, n]) for i in range(i_count)]
        # c08：1 + (1/Φ_i)·Re Tr(H_i B_i)、以及 z_ref/(D_i^off·B/ln2)·z̃_i - ℓ_sen ≤ 0
        self.H_re = [model.parameter("H_re_%d" % i, [n, n]) for i in range(i_count)]
        self.H_im = [model.parameter("H_im_%d" % i, [n, n]) for i in range(i_count)]
        self.c08_inv_phi = model.parameter("c08_inv_phi", i_count)
        self.c08_inv_row_scale = model.parameter("c08_inv_row_scale", i_count)
        # c09：D_j - (D_j^max - STRICT_TOL) ≤ 0
        self.c09_rhs = model.parameter("c09_rhs", j_count)
        # c10：d_slack = D_j^max - D_j；(Σ f + Σ C_jL_j r_j - F_max)/f_scale ≤ 0
        self.c10_d_max = model.parameter("c10_d_max", j_count)
        self.c10_c_l = model.parameter("c10_c_l", j_count)
        self.c10_f_max = model.parameter("c10_f_max")
        self.inv_freq_scale = model.parameter("inv_freq_scale")
        # c12：-dw_j D_j + const_j + (-w1_j/ln2)ℓ_w + cu_k1 Σᵢ η_ij Re Tr(H_i W_i) - cu_p1 ...
        #      η_ij 单独作为标量参数（与旧实现 mul(eta_col[i], ·) 逐字对应）
        self.c12_ndw = model.parameter("c12_ndw", j_count)          # -dw_j
        self.c12_const = model.parameter("c12_const", j_count)      # const_j
        self.c12_nlw = model.parameter("c12_nlw", j_count)          # -w1_j/ln2
        self.c12_nlb = model.parameter("c12_nlb", j_count)          # -w2_j/ln2
        self.c12_eta = [[model.parameter("c12_eta_%d_%d" % (i, j)) for j in range(j_count)]
                        for i in range(i_count)]


class FusionVariables:
    """P5 的 Fusion 变量（复 Hermitian 矩阵拆成实部/虚部两个实矩阵）。"""

    def __init__(self, model, params):
        i_count, j_count, n = params.I, params.J, params.N
        self.I, self.J, self.N = i_count, j_count, n
        self.X = [model.variable("X_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Y = [model.variable("Y_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Xb = [model.variable("Xb_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.Yb = [model.variable("Yb_%d" % i, [n, n], mf.Domain.unbounded()) for i in range(i_count)]
        self.f_norm = model.variable("f_norm", i_count, mf.Domain.greaterThan(0.0))
        # ũ_i = u_ref·un_i、z_i = z_ref·zn_i（归一化变量，量级 O(1)；见 FusionParams）
        self.un = model.variable("un", i_count, mf.Domain.greaterThan(0.0))
        self.zn = model.variable("zn", i_count, mf.Domain.greaterThan(0.0))
        self.D = model.variable("D", j_count, mf.Domain.greaterThan(0.0))
        # 目标 ①② 的锥辅助变量（Fusion 目标必须仿射）
        # Qn 为归一化能耗：Q = q_ref·Qn，使变量量级 O(1)（q_ref 见 FusionParams）
        self.Qn = model.variable("Qn", 1, mf.Domain.greaterThan(0.0))
        self.s = model.variable("s", j_count, mf.Domain.greaterThan(0.0))
        self.r = model.variable("r", j_count, mf.Domain.greaterThan(0.0))
        # c08 / c12 中 log 的下界辅助变量：(·, 1, ℓ) ∈ K_exp ⇔ ℓ ≤ ln(·)
        self.ell_sen = model.variable("ell_sen", i_count, mf.Domain.unbounded())
        self.ell_w = model.variable("ell_w", j_count, mf.Domain.unbounded())
        self.ell_b = model.variable("ell_b", j_count, mf.Domain.unbounded())


def sync_fusion(model, fp, ctx):
    """由当前线性化点与当前场景刷新全部 Fusion Parameter 取值。

    数据来源与旧实现逐项相同：CCCP 线性化量取自 ``cccp_params.compute_values`` /
    ``cu_row_constants``，其余取自 ctx（信道、功率、时延等）。每个 Parameter 的取值
    就是旧实现里对应位置的 numpy 常数，**不额外做任何折叠**（见模块 docstring）。
    """
    I, J = ctx.I, ctx.J
    freq_scale = ctx.freq_scale

    # ── (a) CCCP 线性化量 ─────────────────────────────────────────────────────
    v = compute_values(ctx)
    fp.d_prev_z_u.setValue(_array(v["d_prev_z_u"]))
    fp.cst_prev.setValue(_array(v["cst_prev"]))
    fp.bias_r1.setValue(_array(v["bias_r1"]))
    fp.bias_r1_off.setValue(_array(v["bias_r1_off"]))
    fp.psi_sen_bias.setValue(_array(v["psi_sen_bias"]))
    for i in range(I):
        fp.M_sen_re[i].setValue(_array(v["M_sen_re"][i]))
        fp.M_sen_im[i].setValue(_array(v["M_sen_im"][i]))
        fp.M_off_re[i].setValue(_array(v["M_off_re"][i]))
        fp.M_off_im[i].setValue(_array(v["M_off_im"][i]))
        fp.M_psi_re[i].setValue(_array(v["M_psi_re"][i]))
        fp.M_psi_im[i].setValue(_array(v["M_psi_im"][i]))
    for name in ("cu_r1", "cu_r2", "cu_inv1", "cu_inv2", "cu_k1", "cu_k2", "cu_p1", "cu_p2"):
        getattr(fp, name).setValue(_array(v[name]))

    # ── (b) 归一化参考量：取当前线性化点的量级作为参考 ────────────────────────
    # 与原实现一致：z_ref = max(|z|)、u_ref = max((f/f_scale)²)、q_ref = Σ(z²+ũ²)。
    z_prev = np.asarray(ctx.z_aux_rate_prev, dtype=float).reshape(-1)
    u_prev = (np.asarray(ctx.f_uav_freq_prev, dtype=float).reshape(-1) / freq_scale) ** 2
    z_ref = max(float(np.max(np.abs(z_prev))) if z_prev.size else 0.0, 1.0)
    u_ref = max(float(np.max(np.abs(u_prev))) if u_prev.size else 0.0, 1e-6)
    q_ref = max(float(np.sum(z_prev ** 2 + u_prev ** 2)), 1.0)
    fp.q_ref.setValue(q_ref)
    fp.z_ref.setValue(z_ref)
    fp.u_ref.setValue(u_ref)
    fp.inv_sqrt_u_ref.setValue(1.0 / np.sqrt(u_ref))
    fp.zn_over_sqrt_qref.setValue(z_ref / np.sqrt(q_ref))
    fp.un_over_sqrt_qref.setValue(u_ref / np.sqrt(q_ref))
    fp.z_ref_now = z_ref

    # ── (c) 目标函数系数（与 build_objective_fusion 里的用法一一对应）────────
    omega_1, omega_2, omega_3 = ctx.omega_1, ctx.omega_2, ctx.omega_3
    g_sen = omega_1 * ctx.kappa_cpu * ctx.C_sen * freq_scale ** 2
    fp.obj_g_sen_half.setValue(g_sen / 2.0)
    fp.obj_g_sen.setValue(g_sen)
    fp.obj_neg_g_sen.setValue(-g_sen)
    fp.obj_coeff2.setValue(_array(
        omega_1 * ctx.kappa_cpu * (np.asarray(ctx.C_cu_cycles, dtype=float)
                                   * np.asarray(ctx.L_cu_task, dtype=float)) ** 3))
    fp.obj_omega_2.setValue(omega_2)
    fp.obj_dbar.setValue(ctx.D_bar_sen)
    fp.obj_doff.setValue(_array(ctx.D_uav_off))
    fp.obj_efly.setValue(_array(ctx.E_uav_fly))
    fp.obj_omega_3.setValue(omega_3)
    fp.obj_p_cu.setValue(_array(ctx.p_cu_power))

    # ── (d) 约束系数（逐个等价于旧实现里的那个 numpy 常数）──────────────────
    eta_share = np.asarray(ctx.eta_share, dtype=float)
    D_uav_off = np.asarray(ctx.D_uav_off, dtype=float)
    D_max_cu = np.asarray(ctx.D_max_cu, dtype=float)

    # c01
    fp.c01_rhs.setValue(_array(ctx.D_bar_sen + D_uav_off))
    for i in range(I):
        fp.c01_eta[i].setValue(_array(eta_share[i, :]))

    # c02（旧实现：coeff = D̄^sen + D_i^off - D^sen_max，f_uav = f_scale·f̃）
    fp.c02_coeff.setValue(_array(ctx.D_bar_sen + D_uav_off - ctx.D_max_sen))
    fp.c02_c_sen.setValue(ctx.C_sen)
    fp.freq_scale.setValue(freq_scale)
    fp.inv_freq_scale.setValue(1.0 / freq_scale)

    # c05 / c07
    fp.p_max_uav.setValue(ctx.P_max_uav)

    # c06
    fp.c06_rhs.setValue(_array(ctx.eps_sinr * np.asarray(ctx.Gamma_sinr, dtype=float)))

    # c08 / c12 共用的 H_i(t) 与 G_i(t)
    G_sen_corr = np.asarray(ctx.G_sen_corr, dtype=complex)
    H_uav_bs = np.asarray(ctx.H_uav_bs, dtype=complex)
    Phi_off_inr = np.asarray(ctx.Phi_off_inr, dtype=float)
    for i in range(I):
        fp.G_re[i].setValue(_array(G_sen_corr[i].real))
        fp.G_im[i].setValue(_array(G_sen_corr[i].imag))
        fp.H_re[i].setValue(_array(H_uav_bs[i].real))
        fp.H_im[i].setValue(_array(H_uav_bs[i].imag))
    fp.c08_inv_phi.setValue(_array(1.0 / Phi_off_inr))
    # 旧实现：row_scale = D_i^off·B/ln2，行内用 1/row_scale 缩放 z_ref·z̃_i
    # （严格按旧实现的运算顺序求 1/row_scale，避免最后一位舍入不同）
    fp.c08_inv_row_scale.setValue(_array(1.0 / (D_uav_off * ctx.B / LN2)))

    # c09
    fp.c09_rhs.setValue(_array(D_max_cu - STRICT_TOL))

    # c10
    fp.c10_d_max.setValue(_array(D_max_cu))
    fp.c10_c_l.setValue(_array(np.asarray(ctx.C_cu_cycles, dtype=float)
                               * np.asarray(ctx.L_cu_task, dtype=float)))
    fp.c10_f_max.setValue(ctx.F_max)

    # c12（与 cu_row_constants 同源；η 逐元素单独给参数）
    w1, w2, dw, const, _ = cu_row_constants(ctx)
    fp.c12_ndw.setValue(_array(-dw))
    fp.c12_const.setValue(_array(const))
    fp.c12_nlw.setValue(_array(-w1 / LN2))
    fp.c12_nlb.setValue(_array(-w2 / LN2))
    for i in range(I):
        for j in range(J):
            fp.c12_eta[i][j].setValue(float(eta_share[i, j]))

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


def build_objective_fusion(fp, fv):
    """构造 P5 目标：与 objective.build_objective 逐项对应，非线性项用锥辅助变量替代。

    表达式树与旧实现逐字对应，只把其中的 numpy 常数换成同名参数
    （g_sen / coeff2 / omega_2 / D̄^sen / D_i^off / E_i^fly / omega_3 / p_j）。
    """
    trace_id = np.eye(fv.N)

    terms = []
    # ① 感知计算能耗 (g_sen/2)·Q，Q ≥ Σ(z_i² + ũ_i²)（锥约束见 build_constraints_fusion）
    terms.append(mf.Expr.mul(fp.obj_g_sen_half, mf.Expr.mul(fp.q_ref, fv.Qn.index(0))))

    # ⑤ 乘积 g·z_i·ũ_i 的 CCCP 线性化：- g·Σ d_i(z_i - ũ_i) + g·Σ cst_i
    #   代入 z = z_ref·zn、ũ = u_ref·un（恒等）
    terms.append(mf.Expr.mul(fp.obj_neg_g_sen, mf.Expr.mul(fp.z_ref,
                                                           mf.Expr.dot(fp.d_prev_z_u, fv.zn))))
    terms.append(mf.Expr.mul(fp.obj_g_sen, mf.Expr.mul(fp.u_ref,
                                                       mf.Expr.dot(fp.d_prev_z_u, fv.un))))
    terms.append(mf.Expr.mul(fp.obj_g_sen, mf.Expr.sum(fp.cst_prev)))

    # ② 娱乐计算能耗 Σ ω₁κ(C_jL_j)³·s_j，s_j ≥ 1/(D_j^max-D_j)²（幂锥给出）
    terms.append(mf.Expr.dot(fp.obj_coeff2, fv.s))

    # ③ UAV 感知/卸载/飞行能耗
    for i in range(fv.I):
        per_uav = _sum_terms([
            mf.Expr.mul(fp.obj_dbar, mf.Expr.dot(trace_id, fv.X[i])),
            mf.Expr.mul(fp.obj_doff.index(i), mf.Expr.dot(trace_id, fv.Xb[i])),
            fp.obj_efly.index(i),
        ])
        terms.append(mf.Expr.mul(fp.obj_omega_2, per_uav))

    # ④ CU 卸载能耗
    terms.append(mf.Expr.mul(fp.obj_omega_3, mf.Expr.dot(fp.obj_p_cu, fv.D)))

    # ⑥ 秩一罚项 Re Tr(M_i W_i) + bias_i
    for i in range(fv.I):
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
    """构造全部约束，逐条对应 constraints/c01..c13（系数全部由 Parameter 提供）。

    与旧实现相比，写法逐条对应，唯一的**结构性**改动在 c12：旧实现按数值分支决定是否
    建项（``if eta_col[i] == 0: continue`` / ``if w1[j] > 0`` / ``if w2[j] > 0``），
    模型结构会随时隙变化、无法复用。这里改为**总是**建全部项与两个指数锥，用参数把
    系数置零（η_ij=0 / w1 or w2=0），数值上等价（零系数贡献零），但结构固定。
    """
    I, J, N = ctx.I, ctx.J, ctx.N
    trace_id = np.eye(N)
    f_uav = [mf.Expr.mul(fp.freq_scale, fv.f_norm.index(i)) for i in range(I)]

    # ── c01 Task-Fresh：D̄^sen + D_off - Σ_j η_ij D_j ≤ 0 ────────────────────
    for i in range(I):
        model.constraint("c01_task_fresh_%d" % i,
                         mf.Expr.sub(fp.c01_rhs.index(i), mf.Expr.dot(fp.c01_eta[i], fv.D)),
                         mf.Domain.lessThan(0.0))

    # ── c02 UAV-task-delay：(D̄^sen + D_off - D_sen^max)·f_i + C_sen·z_i ≤ 0 ──
    for i in range(I):
        expr = mf.Expr.add(mf.Expr.mul(fp.c02_coeff.index(i), f_uav[i]),
                           mf.Expr.mul(fp.c02_c_sen,
                                       mf.Expr.mul(fp.z_ref, fv.zn.index(i))))
        model.constraint("c02_uav_task_delay_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c03 / c04 半正定 ─────────────────────────────────────────────────────
    for i in range(I):
        _add_hermitian_psd(model, fv.X[i], fv.Y[i], "sen_%d" % i, N)
        _add_hermitian_psd(model, fv.Xb[i], fv.Yb[i], "off_%d" % i, N)

    # ── c05 UAV-Sen-Max-Power：Tr(W_i) ≤ P_max ──────────────────────────────
    for i in range(I):
        model.constraint("c05_sen_pow_%d" % i,
                         mf.Expr.sub(mf.Expr.dot(trace_id, fv.X[i]), fp.p_max_uav),
                         mf.Domain.lessThan(0.0))

    # ── c06 Sensing-SINR：εΓ_i - Re Tr(G_i W_i) ≤ 0 ─────────────────────────
    for i in range(I):
        expr = mf.Expr.sub(fp.c06_rhs.index(i),
                           _real_trace(fp.G_re[i], fp.G_im[i], fv.X[i], fv.Y[i]))
        model.constraint("c06_sensing_sinr_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c07 UAV-Off-Max-Power：Tr(B_i) ≤ P_max ──────────────────────────────
    for i in range(I):
        model.constraint("c07_off_pow_%d" % i,
                         mf.Expr.sub(mf.Expr.dot(trace_id, fv.Xb[i]), fp.p_max_uav),
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
    for i in range(I):
        arg_y = mf.Expr.add(1.0,
                            mf.Expr.mul(fp.c08_inv_phi.index(i),
                                        _real_trace(fp.H_re[i], fp.H_im[i],
                                                    fv.Xb[i], fv.Yb[i])))
        model.constraint("c08_exp_%d" % i,
                         mf.Expr.vstack(arg_y, 1.0, fv.ell_sen.index(i)),
                         mf.Domain.inPExpCone())
        model.constraint("c08_%d" % i,
                         mf.Expr.sub(
                             mf.Expr.mul(fp.c08_inv_row_scale.index(i),
                                         mf.Expr.mul(fp.z_ref, fv.zn.index(i))),
                             fv.ell_sen.index(i)),
                         mf.Domain.lessThan(0.0))

    # ── c09 CU-Frequency-Always-Positive：D_j - D_j^max + STRICT_TOL ≤ 0 ────
    for j in range(J):
        model.constraint("c09_%d" % j,
                         mf.Expr.sub(fv.D.index(j), fp.c09_rhs.index(j)),
                         mf.Domain.lessThan(0.0))

    # ── c10 BS-Max-Frequency ────────────────────────────────────────────────
    #     r_j ≥ 1/(D_j^max - D_j)（旋转二阶锥），(Σ f + Σ C L r - F_max)/freq_scale ≤ 0
    for j in range(J):
        model.constraint("c10_rcone_%d" % j,
                         mf.Expr.vstack(fv.r.index(j),
                                        mf.Expr.sub(fp.c10_d_max.index(j), fv.D.index(j)),
                                        SQRT2),
                         mf.Domain.inRotatedQCone())
    freq_terms = list(f_uav)
    for j in range(J):
        freq_terms.append(mf.Expr.mul(fp.c10_c_l.index(j), fv.r.index(j)))
    model.constraint("c10_bs_freq",
                     mf.Expr.mul(fp.inv_freq_scale,
                                 mf.Expr.sub(_sum_terms(freq_terms), fp.c10_f_max)),
                     mf.Domain.lessThan(0.0))

    # ── c11 CCCP:Auxiliary-Variable-1：-z_i + Re Tr(M_psi W_i) + bias ≤ 0 ──
    for i in range(I):
        expr = mf.Expr.add(
            mf.Expr.neg(mf.Expr.mul(fp.z_ref, fv.zn.index(i))),
            _real_trace(fp.M_psi_re[i], fp.M_psi_im[i], fv.X[i], fv.Y[i]))
        expr = mf.Expr.add(expr, fp.psi_sen_bias.index(i))
        model.constraint("c11_%d" % i, expr, mf.Domain.lessThan(0.0))

    # ── c12 CCCP:CUs-task-successful-offloading ─────────────────────────────
    #     旧实现按 η_ij 是否为 0 决定是否加入该项；这里总是建全部 I 项，
    #     用 c12_eta[i][j]（=η_ij）作为系数，η_ij=0 时该项恒为 0，数值等价。
    #     已实测：Fusion 组装时会把 0 系数项直接丢掉（两侧任务里非零元个数完全相同，
    #     PTF 逐系数比对最大相对差 0），故"显式零项"不带来任何数值差异。
    #     同一理由下，w1_j / w2_j 为 0 的 CU 也总是建两个指数锥：此时 ℓ_w[j]/ℓ_b[j]
    #     在目标与其它约束里的系数都是 0，该锥只给它们一个上界，不影响最优值。
    for j in range(J):
        terms_w, terms_b = [], []
        for i in range(I):
            terms_w.append(mf.Expr.mul(fp.c12_eta[i][j], _real_trace(
                fp.H_re[i], fp.H_im[i], fv.X[i], fv.Y[i])))
            terms_b.append(mf.Expr.mul(fp.c12_eta[i][j], _real_trace(
                fp.H_re[i], fp.H_im[i], fv.Xb[i], fv.Yb[i])))
        sum_w = _sum_terms(terms_w)
        sum_b = _sum_terms(terms_b)

        model.constraint("c12_exp_w_%d" % j,
                         mf.Expr.vstack(
                             mf.Expr.add(fp.cu_r1.index(j),
                                         mf.Expr.mul(fp.cu_inv1.index(j), sum_w)),
                             1.0, fv.ell_w.index(j)),
                         mf.Domain.inPExpCone())
        model.constraint("c12_exp_b_%d" % j,
                         mf.Expr.vstack(
                             mf.Expr.add(fp.cu_r2.index(j),
                                         mf.Expr.mul(fp.cu_inv2.index(j), sum_b)),
                             1.0, fv.ell_b.index(j)),
                         mf.Domain.inPExpCone())

        expr = mf.Expr.add(mf.Expr.mul(fp.c12_ndw.index(j), fv.D.index(j)),
                           fp.c12_const.index(j))
        expr = mf.Expr.add(expr, mf.Expr.mul(fp.c12_nlw.index(j), fv.ell_w.index(j)))
        expr = mf.Expr.add(expr, mf.Expr.mul(fp.cu_k1.index(j), sum_w))
        expr = mf.Expr.sub(expr, fp.cu_p1.index(j))
        expr = mf.Expr.add(expr, mf.Expr.mul(fp.c12_nlb.index(j), fv.ell_b.index(j)))
        expr = mf.Expr.add(expr, mf.Expr.mul(fp.cu_k2.index(j), sum_b))
        expr = mf.Expr.sub(expr, fp.cu_p2.index(j))
        model.constraint("c12_%d" % j, expr, mf.Domain.lessThan(0.0))

    # ── c13 Epigraph-Tau：square(f_i/freq_scale) - ũ_i ≤ 0，即 f̃_i² ≤ ũ_i ──
    for i in range(I):
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
    for j in range(J):
        model.constraint("obj_s_cone_%d" % j,
                         mf.Expr.vstack(fv.s.index(j), 0.5, fv.r.index(j)),
                         mf.Domain.inRotatedQCone())


def build_fusion_model(ctx):
    """构建并返回 (model, FusionParams, FusionVariables)。

    结构与场景完全解耦（结构与系数全部走 Parameter，见模块 docstring）：只要 (I, J, N)
    不变，建出来的模型对任何 ctx 都适用。常规路径请用 ``get_fusion_model`` 复用同一实例；
    本函数每次都会新建一个 Model（仅用于对比测试 / 需要独立模型的场景）。
    """
    params = ctx.params
    model = mf.Model("P5_fusion")
    fp = FusionParams(model, params)
    fv = FusionVariables(model, params)
    build_constraints_fusion(ctx, model, fp, fv)
    model.objective(mf.ObjectiveSense.Minimize, build_objective_fusion(fp, fv))
    return model, fp, fv


# ── 进程内模型缓存：按 (I, J, N) 复用，避免每个时隙新建 Model ─────────────────
# 详见模块 docstring「模型只建一次（进程内复用）」一节。
_MODEL_CACHE = {}


def get_fusion_model(ctx):
    """返回按 (I, J, N) 复用的 (model, fp, fv)；缓存未命中时才构建。

    求解器容差与模型结构无关，首次建模型时设置一次即可，后续复用无需重复设置。
    """
    key = (int(ctx.I), int(ctx.J), int(ctx.N))
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        model, fp, fv = build_fusion_model(ctx)
        # Fusion 的求解器参数名采用 camelCase（无 MSK_ 前缀），
        # 与 cvxpy 侧 mosek_params 里的 MSK_DPAR_INTPNT_CO_TOL_PFEAS/DFEAS 等价
        model.setSolverParam("intpntCoTolPfeas", ctx.mosek_tol_feas)
        model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)
        cached = (model, fp, fv)
        _MODEL_CACHE[key] = cached
    return cached


def _hermitian_from(x_level, y_level, n):
    """由实部/虚部变量取值还原 Hermitian 矩阵：W = X + jY，再强制 Hermitian。"""
    x = np.asarray(x_level, dtype=float).reshape(n, n)
    y = np.asarray(y_level, dtype=float).reshape(n, n)
    w = x + 1j * y
    return (w + w.conj().T) / 2.0


# 是否复用进程内单模型（True = 正常训练路径）。
# 置 False 时每个时隙新建模型并在结束时 dispose——仅用于与旧实现做 A/B 对比测试。
REUSE_MODEL = True


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

    if REUSE_MODEL:
        model, fp, fv = get_fusion_model(ctx)
    else:
        # 非复用路径：新建模型（并在 finally 中 dispose，仅用于 A/B 对比）
        model, fp, fv = build_fusion_model(ctx)
        model.setSolverParam("intpntCoTolPfeas", ctx.mosek_tol_feas)
        model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)

    try:
        # 先按 x^(0) 写参数，之后每轮只 flushParameters 复用同一模型
        update_linearization_points(ctx)
        sync_fusion(model, fp, ctx)

        for n in range(1, ctx.max_iterations + 1):
            try:
                model.solve()
            except Exception as exc:                              # noqa: BLE001
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
    finally:
        if not REUSE_MODEL:
            # 非复用路径必须显式释放原生 MOSEK Task（Model↔Variable 引用环使
            # 仅靠 del 无法及时触发 __del__），否则内存会随时隙单调增长。
            model.dispose()

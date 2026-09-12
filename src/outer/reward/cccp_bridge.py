"""外层 (MHBPPO) -> 内层 PC3P 桥接。

把外层环境**实时算出**的信道 / 位置 / 动作，装配成内层问题 P5 的求解上下文
（Parameters + Environment + InnerVariables），调用 **MOSEK Fusion 后端**
（fusion_pc3p.run_pc3p_fusion）求解，再把结果折算回 MyReward 需要的 12 元组
（与旧 penalty_based_cccp 返回格式一致，故 my_reward.py 只改导入）。

场景一致性：内层 Environment 完全由外层传入的信道构造（assemble_environment 不做任何
随机采样），因此内层求解与外层 my_env 处于**同一场景**。

变量对应关系（外层 -> 内层）::

    uavs_2_cus_channels[i, j]       -> h_cu_2_uav[i, j]      (I,J,N)
    uavs_2_bs_channels[i, 0]        -> h_uav_2_bs[i]         (I,N)
    cus_2_bs_channels[j, 0]         -> h_cu_2_bs[j]          (J,)
    uavs_2_targets_channels[i, k*]  -> A_theta[i]            (I,N,N)  k* = 指定感知目标
    uavs_cus_matched_matrix         -> eta_share             (I,J)
    cus_off_power                   -> p_cu_power            (J,)
    uavs_off_duration               -> D_uav_off             (I,)
    uavs_pos / uavs_pos_cur         -> q_uav_pos / q_uav_pos_next
    uavs_rec_beam_vectors           -> g_rec_beam            (I,N)

g_rec_beam（UAV 接收波束，‖g_i‖²=1）由**外层动作空间产出**（my_env 追加的
uav_rec_beam_dir_real / uav_rec_beam_dir_imag 动作头，逐 UAV L2 归一化），与其他外层给定
量（eta_share / p_cu_power / D_uav_off）并列注入内层参与能耗计算。调用方未传
``uavs_rec_beam_vectors`` 时，回退取**指定感知目标信道 A_i 的主特征向量**（A_i 秩一分解后的
匹配接收波束 a_i/‖a_i‖），使接收增益 |a_i^H g_i| 最大。
"""

import importlib.util
import os
import sys
from math import log10

import numpy as np

# ── 内层模块加载 ─────────────────────────────────────────────────────────────
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))        # src/outer/reward
_SRC_DIR = os.path.dirname(os.path.dirname(_BRIDGE_DIR))        # src
_INNER_DIR = os.path.join(_SRC_DIR, "inner")
_INNER_ENV_ALIAS = "_iscc_inner_environment"


def _ensure_inner_on_path():
    """把内层目录挂到 sys.path 末尾，供内层模块间的扁平导入（from cccp_params import ...）。

    追加到**末尾**：即便外层存在同名模块也优先用外层已加载版本，避免污染外层命名空间。
    """
    if not os.path.isdir(_INNER_DIR):
        raise ImportError("未找到内层目录：{}".format(_INNER_DIR))
    if _INNER_DIR not in sys.path:
        sys.path.append(_INNER_DIR)


def _load_inner_environment():
    """按文件路径加载内层 environment.py。

    内层 environment.py 与外层 environment **包**同名，直接 import environment 会拿到外层
    包（已在 sys.modules 中），故用 importlib 以独立模块名加载。该模块只依赖 dataclasses /
    numpy，可独立加载；内层其余模块也不依赖它。
    """
    if _INNER_ENV_ALIAS in sys.modules:
        return sys.modules[_INNER_ENV_ALIAS]
    spec = importlib.util.spec_from_file_location(
        _INNER_ENV_ALIAS, os.path.join(_INNER_DIR, "environment.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_INNER_ENV_ALIAS] = module
    spec.loader.exec_module(module)
    return module


_ensure_inner_on_path()
from parameters import Parameters                                    # noqa: E402
from variables import InnerVariables, InnerContext                   # noqa: E402
from pc3p import compute_pure_energy                                 # noqa: E402
# 求解后端：MOSEK Fusion（fusion_pc3p）——与 cvxpy 后端共用同一套线性化量，
# 返回的 result 结构完全一致；此处以别名 run_pc3p 导入，调用点无需改动。
from fusion_pc3p import run_pc3p_fusion as run_pc3p                  # noqa: E402

_assemble_environment = _load_inner_environment().assemble_environment


# ── 小工具 ───────────────────────────────────────────────────────────────────
def _pick(base_args, outer_name, default):
    """取外层参数，缺失时回落到内层默认值（对 base_args 字段变动有韧性）。"""
    value = getattr(base_args, outer_name, None)
    return default if value is None else value


def _watt_2_dbm(watt):
    return 10.0 * log10(watt) + 30.0


def _matched_receive_beam(A):
    """A 的主特征向量（A = c·a a^H 时即 a/‖a‖），作为指定目标的匹配接收波束。"""
    hermitian = (A + A.conj().T) / 2.0
    _, eigenvectors = np.linalg.eigh(hermitian)
    beam = eigenvectors[:, -1]
    norm = np.linalg.norm(beam)
    return beam / norm if norm > 0 else beam


def _distance_from_sensing_matrix(A, params):
    """由感知信道反解 UAV 到指定目标距离。

    A = sqrt(ξ_0 ρ d^{-4}) · a a^H 且 ‖a a^H‖_F = N，故 ‖A‖_F = sqrt(ξ_0 ρ d^{-4})·N，
    于是 d = ( ξ_0 ρ / (‖A‖_F / N)² )^{1/4}。用于填充 Environment.d_uav_target。
    """
    path_gain = np.linalg.norm(A, "fro") / params.N
    return float((params.xi_0 * params.rho_ref / max(path_gain ** 2, 1e-300)) ** 0.25)


# ── 参数映射 ─────────────────────────────────────────────────────────────────
def build_inner_parameters(base_args, cus_entertaining_task_size=None):
    """把外层 base_args 映射为内层 Parameters。

    只对齐**物理场景参数**（必须与 my_env 用的一致）；PC3P 求解器超参（gamma_1/2、
    rho_penalty*、max_iterations、freq_scale、mosek_tol_feas）沿用内层论文默认值，
    因为那是新求解器调好的口径。
    """
    d = Parameters()
    uav_max_power_w = getattr(base_args, "uav_max_power", None)
    overrides = {
        # 场景规模
        "I": _pick(base_args, "uavs_num", d.I),
        "J": _pick(base_args, "cus_num", d.J),
        "K": _pick(base_args, "targets_num", d.K),
        "deploy_radius": _pick(base_args, "radius", d.deploy_radius),
        "area_length": _pick(base_args, "radius", d.area_length),
        "uav_height": _pick(base_args, "uav_height", d.uav_height),
        # UAV 飞行
        "uav_max_speed": _pick(base_args, "uav_max_speed", d.uav_max_speed),
        "uav_min_speed": _pick(base_args, "uav_min_speed", d.uav_min_speed),
        "varrho_1": _pick(base_args, "uav_c1", d.varrho_1),
        "varrho_2": _pick(base_args, "uav_c2", d.varrho_2),
        "uav_safe_distance": _pick(base_args, "uav_safe_distance", d.uav_safe_distance),
        # 信道
        "B": _pick(base_args, "bandwidth", d.B),
        "noise_power_density_dbm": _pick(base_args, "noise_power_density_dbm",
                                         d.noise_power_density_dbm),
        "ref_path_loss_db": _pick(base_args, "ref_path_loss_db", d.ref_path_loss_db),
        "rician_factor_db": _pick(base_args, "rician_factor_db", d.rician_factor_db),
        "alpha_1": _pick(base_args, "alpha_uav_link", d.alpha_1),
        "alpha_2": _pick(base_args, "alpha_uav_link", d.alpha_2),
        "alpha_3": _pick(base_args, "alpha_cu_link", d.alpha_3),
        "N": _pick(base_args, "antenna_nums", d.N),
        "d_over_lambda": _pick(base_args, "frac_d_lambda", d.d_over_lambda),
        # 感知
        "xi_0": _pick(base_args, "radar_rcs", d.xi_0),
        "eps_sinr_db": _pick(base_args, "sen_sinr", d.eps_sinr_db),
        "delta_radar": _pick(base_args, "radar_duty_ratio", d.delta_radar),
        "sigma_pre_sq": _pick(base_args, "var_range_fluctuation", d.sigma_pre_sq),
        "nu_pulse": _pick(base_args, "radar_impulse_duration", d.nu_pulse),
        "gamma_radar": _pick(base_args, "radar_spectrum_shape", d.gamma_radar),
        "D_bar_sen": _pick(base_args, "uav_sen_duration", d.D_bar_sen),
        # BS 计算
        "C_cycles_per_bit": _pick(base_args, "bs_cycles_per_bit", d.C_cycles_per_bit),
        "F_max": _pick(base_args, "bs_max_freq", d.F_max),
        "kappa_cpu": _pick(base_args, "kappa", d.kappa_cpu),
        # 功率 / 时延
        "P_max_uav_dbm": (_watt_2_dbm(uav_max_power_w) if uav_max_power_w
                          else d.P_max_uav_dbm),
        "P_max_cu_dbm": _pick(base_args, "cu_max_power_dbm", d.P_max_cu_dbm),
        "D_max_sen": _pick(base_args, "uav_max_delay", d.D_max_sen),
        "D_max_cu_delay": _pick(base_args, "cu_max_delay", d.D_max_cu_delay),
        "tau_slot": _pick(base_args, "time_slot_duration", d.tau_slot),
        # 权重
        "omega_1": _pick(base_args, "omega_weight_1", d.omega_1),
        "omega_2": _pick(base_args, "omega_weight_2", d.omega_2),
        "omega_3": _pick(base_args, "omega_weight_3", d.omega_3),
        "seed": _pick(base_args, "seed", d.seed),
    }
    params = Parameters(**overrides)
    # CU 娱乐任务数据量：外层逐 CU 给出，覆盖 Parameters 中由 L_cu_task_bits 铺平的结果
    if cus_entertaining_task_size is not None:
        params.L_cu_task = np.asarray(cus_entertaining_task_size, dtype=float).reshape(-1)
    return params


# ── 主入口 ───────────────────────────────────────────────────────────────────
def solve_inner_energy(args, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels,
                       uavs_2_targets_channels, uavs_targets_matched_matrix,
                       uavs_cus_matched_matrix, uavs_pos_pre, uavs_pos_cur,
                       uavs_off_duration, cus_off_power, cus_entertaining_task_size,
                       uavs_rec_beam_vectors=None, return_solution=True):
    """外层场景 -> 内层 PC3P 求解 -> 返回与旧 penalty_based_cccp 一致的 12 元组。

    :param uavs_rec_beam_vectors: 可选，(I, N) 复数，外层智能体产出的接收波束 g_i
        （‖g_i‖²=1）。为 None 时回退到指定感知目标信道 A_i 的匹配接收波束。
    返回 (energy_opt, _, _, _, _, _, _, per_uav_sen_power_list, per_uav_off_power_list,
          per_uav_bs_freq_list, cur_cus_off_duration, solution_payload)；
    无可行解时 energy_opt = inf，各明细为空。
    """
    params = build_inner_parameters(args, cus_entertaining_task_size)
    I, J, N = params.I, params.J, params.N
    num_targets = int(np.asarray(uavs_2_targets_channels).shape[1])

    q_uav_pos = np.asarray(uavs_pos_pre, dtype=float).reshape(I, 3)
    q_uav_pos_next = np.asarray(uavs_pos_cur, dtype=float).reshape(I, 3)

    # ── 信道：对齐内层索引约定 ───────────────────────────────────────────────
    h_cu_2_uav = np.asarray(uavs_2_cus_channels, dtype=complex)               # (I, J, N)
    h_uav_2_bs = np.asarray(uavs_2_bs_channels, dtype=complex).reshape(I, N)  # (I, 1, N)
    h_cu_2_bs = np.asarray(cus_2_bs_channels, dtype=complex).reshape(J)       # (J, 1)

    # ── A(θ)：取每个 UAV 的指定感知目标（匹配矩阵按行 one-hot） ──────────────
    matched = np.asarray(uavs_targets_matched_matrix, dtype=float).reshape(I, num_targets)
    target_idx = np.argmax(matched, axis=1)
    A_all = np.asarray(uavs_2_targets_channels, dtype=complex)                # (I, K, N, N)
    A_theta = np.stack([A_all[i, target_idx[i]] for i in range(I)], axis=0)

    # ── 外层给定量 ───────────────────────────────────────────────────────────
    # g_i：UAV 接收波束（由外层动作空间产出，‖g_i‖²=1）；未提供时回退匹配接收波束
    if uavs_rec_beam_vectors is None:
        g_rec_beam = np.stack([_matched_receive_beam(A_theta[i]) for i in range(I)], axis=0)
    else:
        g_rec_beam = np.asarray(uavs_rec_beam_vectors, dtype=complex).reshape(I, N)
        g_norm = np.linalg.norm(g_rec_beam, axis=1, keepdims=True)
        g_rec_beam = g_rec_beam / np.where(g_norm > 0, g_norm, 1.0)
    eta_share = np.asarray(uavs_cus_matched_matrix, dtype=float).reshape(I, J)
    p_cu_power = np.asarray(cus_off_power, dtype=float).reshape(-1)
    D_uav_off = np.asarray(uavs_off_duration, dtype=float).reshape(-1)

    # ── 由 A_i 反解 d_i（A_i 已含双程路损，直接给 assemble 用） ──────────────
    d_uav_target = np.array([_distance_from_sensing_matrix(A_theta[i], params)
                             for i in range(I)])

    environment = _assemble_environment(
        params,
        q_uav_pos=q_uav_pos,
        q_uav_pos_next=q_uav_pos_next,
        # CU / 目标绝对位置在外层 reward 入参中不可得，且内层除 environment 本身外无人使用，
        # 故以占位填零；d_uav_target 已由上一步显式给出，不受占位影响。
        q_cu_pos=np.zeros((J, 3)),
        q_target_pos=np.zeros((num_targets, 3)),
        q_target_designated=q_uav_pos.copy(),
        h_cu_2_uav=h_cu_2_uav,
        h_uav_2_bs=h_uav_2_bs,
        h_cu_2_bs=h_cu_2_bs,
        A_theta=A_theta,
        g_rec_beam=g_rec_beam,
        eta_share=eta_share,
        p_cu_power=p_cu_power,
        D_uav_off=D_uav_off,
        d_uav_target=d_uav_target,
    )

    variables = InnerVariables.create(params)
    ctx = InnerContext(params, environment, variables)
    result = run_pc3p(ctx)

    if result["W_sen_beam"] is None:
        return (float("inf"), None, None, None, None, None, None,
                [], [], [], np.array([]), None)

    W_sen = np.asarray(result["W_sen_beam"])
    B_off = np.asarray(result["B_off_beam"])
    f_uav_freq = np.asarray(result["f_uav_freq"], dtype=float)
    z_aux_rate = np.asarray(result["z_aux_rate"], dtype=float)
    D_cu_off = np.asarray(result["D_cu_off"], dtype=float)

    energy_opt = compute_pure_energy(ctx, W_sen, B_off, f_uav_freq, z_aux_rate, D_cu_off)

    per_uav_sen_power_list = [float(np.real(np.trace(W_sen[i]))) for i in range(I)]
    per_uav_off_power_list = [float(np.real(np.trace(B_off[i]))) for i in range(I)]
    per_uav_bs_freq_list = [float(f_uav_freq[i]) for i in range(I)]

    # auxiliary_variable_z：旧接口约定为"按 z_scale 归一化后的 z"（my_reward 会再乘 z_scale
    # 还原），此处做同样换算以保持原有能耗分解口径；原始 bit 值放在 z_aux_rate_bits。
    z_scale = float(getattr(args, "z_scale", 1e5))
    solution_payload = {
        "auxiliary_variable_z": z_aux_rate / z_scale,
        "z_aux_rate_bits": z_aux_rate,
    }

    return (energy_opt, None, None, None, None, None, None,
            per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list,
            D_cu_off, solution_payload)

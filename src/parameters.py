"""论文仿真参数（tex/thesis.tex, Section \ref{sec: Simulation Setup}）。

所有参数严格取自论文的 Simulation Setup 小节，单位统一为 SI（W、Hz、bit、s、m）。
"""

from dataclasses import dataclass

import numpy as np
from math import pi, sqrt


def db_2_linear(value_db):
    """dB -> 线性值（功率/幅度类的 10^(x/10) 换算）。"""
    return 10.0 ** (value_db / 10.0)


def dbm_2_watt(value_dbm):
    """dBm -> W。"""
    return 10.0 ** (value_dbm / 10.0) / 1000.0


@dataclass
class Parameters:
    # ── 场景规模 ──────────────────────────────────────────────────────────
    area_length: float = 600.0          # 仿真区域边长 (m)，以 (0, 0, 0) 为中心
    bs_pos: tuple = (0.0, 0.0, 20.0)    # BS 位置
    deploy_radius: float = 600.0        # UAV/CU/target 随机部署圆半径 (m)，以 BS 为圆心
    I: int = 4                          # UAV 数量
    J: int = 10                         # CU 数量
    K: int = 40                         # target 数量

    # ── UAV 飞行参数 ──────────────────────────────────────────────────────
    uav_height: float = 50.0            # H，UAV 固定飞行高度 (m)
    uav_max_speed: float = 40.0         # V_max (m/s)
    uav_min_speed: float = 5.0          # V_min (m/s)
    varrho_1: float = 0.00614           # ϱ_1，飞行能耗参数
    varrho_2: float = 15.976            # ϱ_2，飞行能耗参数
    uav_safe_distance: float = 5.0      # d_min^UAV (m)

    # ── 信道参数 ──────────────────────────────────────────────────────────
    B: float = 10e6                     # 带宽 (Hz)
    noise_power_density_dbm: float = -174.0   # 噪声功率谱密度 (dBm/Hz)
    ref_path_loss_db: float = -30.0     # 1 m 参考路径损耗 ρ (dB)
    rician_factor_db: float = 10.0      # Rician 因子 κ (dB)
    alpha_1: float = 2.0                # CU -> UAV 链路路径损耗因子
    alpha_2: float = 2.0                # UAV -> BS 链路路径损耗因子
    alpha_3: float = 2.5                # CU -> BS 链路路径损耗因子
    N: int = 6                          # UAV 天线数
    d_over_lambda: float = 0.5          # 𝔡/λ，天线间距与波长之比

    # ── 感知参数 ──────────────────────────────────────────────────────────
    xi_0: float = 10.0                  # ξ_0，目标雷达截面积 RCS (m^2)
    eps_sinr_db: float = 20.0           # ε，感知 SINR 门限 (dB)
    delta_radar: float = 1e-2           # δ，雷达占空比
    sigma_pre_sq: float = 1e-14         # σ_pre^2，距离起伏过程方差
    nu_pulse: float = 2e-5              # ν，雷达脉冲持续时间 (s)
    gamma_radar: float = pi / sqrt(3)   # γ，雷达频谱形状参数
    D_bar_sen: float = 0.1              # D̄^sen，感知时长 (s)

    # ── BS 计算参数 ───────────────────────────────────────────────────────
    C_cycles_per_bit: float = 1e3        # C^sen = C_j，处理 1 bit 所需 CPU 周期数
    F_max: float = 10e9                  # F_max，BS 最大计算能力 (Hz)
    kappa_cpu: float = 1e-28             # κ，BS CPU 有效开关电容

    # ── 任务与时延参数 ────────────────────────────────────────────────────
    P_max_uav_dbm: float = 40.0          # P^max_UAV (dBm)
    P_max_cu_dbm: float = 23.0           # P^max_CU (dBm)
    D_max_sen: float = 0.2               # D^sen_max，感知任务最大容忍时延 (s)
    D_max_cu_delay: float = 0.6          # D^max_j，CU 娱乐任务最大容忍时延 (s)
    L_cu_task_bits: float = 170e3        # L_j，CU 娱乐任务数据量 (bits)
    varkappa: int = 4                    # ϰ，一个感知窗口包含的时隙数
    tau_slot: float = 0.6                # τ，时隙长度 (s)

    # ── 求解器数值尺度（仅改变变量的度量单位，不改变任何公式） ────────────
    freq_scale: float = 1e6             # f_{u_i}(t) 以 MHz 为内部单位（1 单位 = 1e6 Hz）

    # ── 权重与算法参数 ────────────────────────────────────────────────────
    omega_1: float = 0.2                 # ω_1，BS 能耗权重
    omega_2: float = 0.4                 # ω_2，UAV 能耗权重
    omega_3: float = 0.4                 # ω_3，CU 能耗权重
    gamma_1: float = 1e-6                # γ_1，PC3P 目标函数收敛阈值
    gamma_2: float = 1e-6                # γ_2，PC3P 秩一约束收敛阈值
    rho_penalty: float = 0.1             # ρ，秩一罚因子
    max_iterations: int = 30             # PC3P 最大迭代次数

    # ── 罚因子递增（P1：秩一间隙未达标时把 ρ 逐步放大）─────────────────────
    # ρ 的初值 0.1 相对目标量级（~1e4）过小，罚项在求解器相对容差下不可见，
    # 导致秩一间隙停在 1e-2~1e-4（远高于 γ₂）。按倍率递增可把间隙压到 1e-6 以下。
    rho_penalty_scale: float = 2.0       # 递增倍率 ρ ← rho_penalty_scale · ρ
    rho_penalty_max: float = 1e5         # ρ 上限

    # ── MOSEK 内点法容差（只改求解器收敛判据，不改动模型，用于缩短单次求解时间）──
    # 实测：把可行性容差由默认 1e-8 放宽到 1e-7，内点迭代数从 34~148 稳定到 ~31，
    # 单次求解耗时约减半，而解与秩一间隙几乎不变（相对误差 ~1e-5）。
    mosek_tol_feas: float = 1e-7

    # ── 随机种子（外层变量尚未由 MHBPPO 给出，当前随机生成） ─────────────
    seed: int = 42

    def __post_init__(self):
        # ── dB -> 线性换算 ────────────────────────────────────────────────
        self.rho_ref = db_2_linear(self.ref_path_loss_db)          # ρ
        self.kappa_rician = db_2_linear(self.rician_factor_db)     # κ（Rician 因子）
        self.sigma_2 = dbm_2_watt(self.noise_power_density_dbm) * self.B  # σ^2 (W)
        self.P_max_uav = dbm_2_watt(self.P_max_uav_dbm)            # P^max_UAV (W)
        self.P_max_cu = dbm_2_watt(self.P_max_cu_dbm)              # P^max_CU (W)
        self.eps_sinr = db_2_linear(self.eps_sinr_db)              # ε

        # ── 感知相关常数 ξ_1、ξ_2 ─────────────────────────────────────────
        self.xi_1 = self.D_bar_sen * self.delta_radar / (2.0 * self.nu_pulse)
        self.xi_2 = (2.0 * self.sigma_pre_sq
                     * (self.gamma_radar ** 2)
                     * (self.B ** 3)
                     * self.nu_pulse)

        # 上镜图变量 τ_i 的尺度（τ_i = z_i + f_{u_i}²，故取 freq_scale²）
        self.tau_scale = self.freq_scale ** 2

        # ── 每 CU 的任务参数 ──────────────────────────────────────────────
        self.C_sen = self.C_cycles_per_bit                          # C^sen
        self.C_cu_cycles = np.full(self.J, self.C_cycles_per_bit)   # C_j(t)
        self.L_cu_task = np.full(self.J, self.L_cu_task_bits)       # L_j(t)
        self.D_max_cu = np.full(self.J, self.D_max_cu_delay)        # D^max_j(t)

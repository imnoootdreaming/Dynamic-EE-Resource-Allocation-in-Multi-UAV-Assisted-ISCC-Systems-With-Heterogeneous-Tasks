"""内层问题 P5 的优化变量与求解上下文。

优化变量（论文 P5 的 var. 部分）：
    W_i(t) = w_i w_i^H  —— UAV 感知波束成形矩阵
    B_i(t) = b_i b_i^H  —— UAV 卸载波束成形矩阵
    f_{u_i}(t)          —— BS 分配给感知任务的计算资源
    z_i(t)              —— 辅助变量（感知数据量上界）
    D_{c_j}^{off}(t)    —— CU 娱乐任务卸载时长
    τ_i                 —— DCP 辅助变量：τ_i ≥ z_i + f_{u_i}^2

线性化点 x^(n)（CCCP 第 n 轮迭代使用，numpy 常量，每轮原地更新）：
    W_sen_beam_prev / B_off_beam_prev / f_uav_freq_prev / z_aux_rate_prev
    Psi_sen / Psi_cu_sen / Psi_cu_off
    nu_max_sen / nu_max_off、W_sen_beam_norm_prev / B_off_beam_norm_prev
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cccp_params import CccpParams


@dataclass
class InnerVariables:
    # ── 优化变量 ──────────────────────────────────────────────────────────
    W_sen_beam: list            # W_i(t) ∈ C^{N×N}，Hermitian PSD
    B_off_beam: list            # B_i(t) ∈ C^{N×N}，Hermitian PSD
    f_uav_freq: cp.Variable     # f_{u_i}(t)          (I,)
    z_aux_rate: cp.Variable     # z_i(t)              (I,)
    D_cu_off: cp.Variable       # D_{c_j}^{off}(t)    (J,)
    tau_zf: cp.Variable         # τ_i，DCP 辅助变量，以 tau_scale 为度量单位  (I,)

    # ── CCCP 线性化点 x^(n) ───────────────────────────────────────────────
    W_sen_beam_prev: np.ndarray        # W_i^{(n)}(t)          (I, N, N)
    B_off_beam_prev: np.ndarray        # B_i^{(n)}(t)          (I, N, N)
    f_uav_freq_prev: np.ndarray        # f_{u_i}^{(n)}(t)      (I,)
    z_aux_rate_prev: np.ndarray        # z_i^{(n)}(t)          (I,)
    D_cu_off_prev: np.ndarray          # D_{c_j}^{off(n)}(t)   (J,)
    Psi_sen: np.ndarray                # Ψ_i^{(n)}(t)          (I,)
    Psi_cu_sen: np.ndarray             # Ψ_{j,1}^{(n)}(t)      (J,)
    Psi_cu_off: np.ndarray             # Ψ_{j,2}^{(n)}(t)      (J,)
    nu_max_sen: np.ndarray             # ν_max(W_i^{(n)}) ν_max^H   (I, N, N)
    nu_max_off: np.ndarray             # ν_max(B_i^{(n)}) ν_max^H   (I, N, N)
    W_sen_beam_norm_prev: np.ndarray   # ‖W_i^{(n)}‖_2         (I,)
    B_off_beam_norm_prev: np.ndarray   # ‖B_i^{(n)}‖_2         (I,)

    @classmethod
    def create(cls, params):
        """按 params 中的 I / J / N 创建全部优化变量与线性化点存储。

        f_{u_i}(t) 与 τ_i 的内部变量带度量单位（f_uav_freq = freq_scale · f_uav_freq_norm，
        tau_zf = tau_scale · tau_zf_norm），这是单纯的度量单位选择，用于改善 MOSEK 的数值条件
        （f 约 10^6 Hz、τ 约 10^13），公式本身保持不变。
        """
        I, J, N = params.I, params.J, params.N
        f_uav_freq_norm = cp.Variable(I, nonneg=True)
        tau_zf_norm = cp.Variable(I, nonneg=True)
        return cls(
            W_sen_beam=[cp.Variable((N, N), hermitian=True) for _ in range(I)],
            B_off_beam=[cp.Variable((N, N), hermitian=True) for _ in range(I)],
            f_uav_freq=params.freq_scale * f_uav_freq_norm,
            z_aux_rate=cp.Variable(I, nonneg=True),
            D_cu_off=cp.Variable(J, nonneg=True),
            tau_zf=tau_zf_norm,
            W_sen_beam_prev=np.zeros((I, N, N), dtype=complex),
            B_off_beam_prev=np.zeros((I, N, N), dtype=complex),
            f_uav_freq_prev=np.zeros(I),
            z_aux_rate_prev=np.zeros(I),
            D_cu_off_prev=np.zeros(J),
            Psi_sen=np.zeros(I),
            Psi_cu_sen=np.zeros(J),
            Psi_cu_off=np.zeros(J),
            nu_max_sen=np.zeros((I, N, N), dtype=complex),
            nu_max_off=np.zeros((I, N, N), dtype=complex),
            W_sen_beam_norm_prev=np.zeros(I),
            B_off_beam_norm_prev=np.zeros(I),
        )


class InnerContext:
    """求解上下文：聚合 parameters / environment / variables。

    约束文件与目标函数文件统一通过 ctx 访问参数、状态与变量，
    例如 ctx.Gamma_sinr、ctx.W_sen_beam、ctx.eps_sinr。
    """

    def __init__(self, params, environment, variables):
        self.params = params
        self.environment = environment
        self.variables = variables
        # CCCP 线性化量的 cvxpy Parameter 集合：使 P5 只编译一次，后续求解复用编译缓存
        self.ccp = CccpParams.create(params)

    def __getattr__(self, name):
        for source in ("variables", "environment", "params"):
            obj = self.__dict__.get(source)
            if obj is not None and hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(name)

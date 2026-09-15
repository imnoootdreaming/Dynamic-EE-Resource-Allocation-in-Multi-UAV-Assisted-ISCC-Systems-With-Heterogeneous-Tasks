"""外层信道模型与功率单位换算。

从 `environment/my_env.py` 抽离的、与环境状态机无关的纯计算函数：

- 单位换算：`db_2_watt` / `dbm_2_watt`
- 通信信道：`compute_com_channel_gain`（Rician 小尺度衰落，支持 MIMO / 非 MIMO，
  可复用预生成的时隙级 NLoS 高斯分量）
- 感知信道：`compute_sen_channel_gain`（UAV→目标双程路损 + 导向矢量外积）

实现与数值逻辑完全取自原 `MyEnv` 方法，仅去掉 `self` 依赖（所需参数全部显式传入），
不改变任何公式、维度或采样方式。
"""

import numpy as np


def dbm_2_watt(dbm):
    return 10 ** ((dbm - 30) / 10)


def db_2_watt(db):
    return 10 ** (db / 10)


def compute_com_channel_gain(uavs_pos, cus_pos, ref_path_loss, frac_d_lambda,
                             alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums,
                             nlos_components=None):
    """计算 UAV-CU / UAV-BS / CU-BS 三类通信链路信道。

    :param uavs_pos: (I, 3) UAV 位置
    :param cus_pos: (J, 3) CU 位置
    :param ref_path_loss: 1m 参考路径损耗（线性值）
    :param frac_d_lambda: 天线间距与波长之比
    :param alpha_uav_link: UAV 链路路径损耗指数
    :param alpha_cu_link: CU 链路路径损耗指数
    :param rician_factor: Rician 因子 K（线性值）
    :param antenna_nums: UAV 天线数 N
    :param nlos_components: 可选，预生成的时隙级 NLoS 高斯分量字典
        （键 `uavs_2_cus` / `uavs_2_bs` / `cus_2_bs`）；未提供时在线采样。
    :return: (uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels)
    """
    bs_pos = np.array([0, 0, 0])

    def get_rician_channel(pos1, pos2, alpha, K, is_mimo=True, nlos_component=None):
        diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        path_loss = ref_path_loss * (dist ** -alpha)

        if is_mimo:
            dx = diff[..., 0]
            dy = diff[..., 1]
            phi = np.arctan2(dy, dx)
            n_range = np.arange(antenna_nums)
            exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(phi)[..., np.newaxis] * n_range
            h_los = np.exp(exponent)
            # 20260404 - NLoS 分量: 优先复用预生成的时隙级高斯样本，仅在未提供时退回在线采样
            if nlos_component is None:
                h_nlos = (
                    np.random.randn(*dist.shape, antenna_nums)
                    + 1j * np.random.randn(*dist.shape, antenna_nums)
                ) / np.sqrt(2)
            else:
                h_nlos = nlos_component
            path_loss_expanded = path_loss[..., np.newaxis]
            h = np.sqrt(path_loss_expanded) * (
                np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos
            )
        else:
            # 20260404 - NLoS 分量: 优先复用预生成的时隙级高斯样本，仅在未提供时退回在线采样
            if nlos_component is None:
                h_nlos = (np.random.randn(*dist.shape) + 1j * np.random.randn(*dist.shape)) / np.sqrt(2)
            else:
                h_nlos = nlos_component
            h = np.sqrt(path_loss) * (
                np.sqrt(K / (K + 1)) * 1.0 + np.sqrt(1 / (K + 1)) * h_nlos
            )
        return h

    uavs_2_cus_nlos = None if nlos_components is None else nlos_components.get("uavs_2_cus")
    uavs_2_bs_nlos = None if nlos_components is None else nlos_components.get("uavs_2_bs")
    cus_2_bs_nlos = None if nlos_components is None else nlos_components.get("cus_2_bs")

    uavs_2_cus_channels = get_rician_channel(
        uavs_pos, cus_pos, alpha_uav_link, rician_factor, is_mimo=True, nlos_component=uavs_2_cus_nlos
    )
    uavs_2_bs_channels = get_rician_channel(
        uavs_pos, bs_pos[np.newaxis, :], alpha_uav_link, rician_factor, is_mimo=True, nlos_component=uavs_2_bs_nlos
    )
    cus_2_bs_channels = get_rician_channel(
        cus_pos, bs_pos[np.newaxis, :], alpha_cu_link, rician_factor, is_mimo=False, nlos_component=cus_2_bs_nlos
    )
    return uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels


def compute_sen_channel_gain(radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
    """计算 UAV→目标的感知信道矩阵 A(θ)（含双程路损）。

    :return: (I, K, N, N) complex ndarray
    """
    diff = uavs_pos[:, np.newaxis, :] - targets_pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dx = diff[..., 0]
    dy = diff[..., 1]
    theta = np.arctan2(dy, dx)

    n_range = np.arange(antenna_nums)
    exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(theta)[..., np.newaxis] * n_range
    a_vec = np.exp(exponent)
    path_gain_amplitude = np.sqrt(radar_rcs * ref_path_loss * (dist ** -4))

    a_vec_col = a_vec[..., np.newaxis]
    a_vec_row_conj = np.conj(a_vec)[..., np.newaxis, :]
    matrix_term = np.matmul(a_vec_col, a_vec_row_conj)
    return path_gain_amplitude[..., np.newaxis, np.newaxis] * matrix_term

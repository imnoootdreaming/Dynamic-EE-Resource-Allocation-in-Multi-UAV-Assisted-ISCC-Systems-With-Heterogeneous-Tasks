import argparse
from math import pi, sqrt
import os
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
from datetime import datetime  # 新增
import csv  # 新增：用于将迭代数据写入 CSV 文件
# from gaussian_based_cccp_algorithm import gaussian_based_cccp, save_and_plot_gaussian_cccp_history
from scipy.optimize import linear_sum_assignment  # 引入线性求和分配函数
# from plot_uav_sensing_beam_pattern import plot_uav_sensing_beam_patterns
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('error', category=RuntimeWarning)  # 把 RuntimeWarning 当作异常处理
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 新增


def get_base_args():
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=30, help="随机案例数量")
    base_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    base_parser.add_argument("--targets_num", type=int, default=40, help="目标数量")
    base_parser.add_argument("--uavs_num", type=int, default=4, help="UAV 数量")
    base_parser.add_argument("--cus_num", type=int, default=10, help="CU 数量")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV 高度 (m)")
    base_parser.add_argument("--radius", type=float, default=600, help="区域半径 (m)")

    # 信道参数
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="1m 参考路径损耗 (dB)")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="天线间距与波长比例")
    base_parser.add_argument("--alpha_uav_link", type=float, default=2, help="UAV 链路路径损耗指数")
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5, help="CU 链路路径损耗指数")
    base_parser.add_argument("--rician_factor_db", type=float, default=10, help="Rician 因子 (dB)")
    base_parser.add_argument("--antenna_nums", type=int, default=6, help="UAV 天线数量")
    base_parser.add_argument("--radar_rcs", type=float, default=10, help="雷达 RCS (mm^2)")
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174, help="噪声功率谱密度 (dBm/Hz)")
    base_parser.add_argument("--bandwidth", type=float, default=10e6, help="带宽 (Hz)")

    # 算法/物理参数
    base_parser.add_argument("--uav_c1", type=float, default=0.00614, help="UAV 飞行参数 c1")
    base_parser.add_argument("--uav_c2", type=float, default=15.976, help="UAV 飞行参数 c2")
    base_parser.add_argument("--kappa", type=float, default=1e-28, help="BS CPU 有效开关电容")
    base_parser.add_argument("--bs_max_freq", type=float, default=10e9, help="BS 最大工作频率 (Hz)")
    base_parser.add_argument("--freq_scale", type=float, default=1e9, help="频率归一化尺度")
    base_parser.add_argument("--z_scale", type=float, default=1e5, help="z 变量归一化尺度")  
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS 处理 1bit 需要的周期数")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.6, help="时隙长度 (s)")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV 感知时长 (s)")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="CU 最大发射功率 (dBm)")
    base_parser.add_argument("--uav_max_power", type=float, default=10, help="UAV 最大功率 (W) = 40dBm")
    base_parser.add_argument("--cu_max_delay", type=float, default=0.6, help="娱乐任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_delay", type=float, default=0.2, help="感知任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_speed", type=float, default=40.0, help="UAV 最大移动速度 (m/s)")
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0, help="UAV 最小移动速度 (m/s)")
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0, help="UAV 安全距离 (m)")
    base_parser.add_argument("--sen_sinr", type=float, default=20, help="感知门限阈值 (dB)")

    # 权重参数
    base_parser.add_argument("--omega_weight_1", type=float, default=0.2, help="BS 权重")
    base_parser.add_argument("--omega_weight_2", type=float, default=0.4, help="UAV 权重")
    base_parser.add_argument("--omega_weight_3", type=float, default=0.4, help="CU 权重")

    # 雷达参数
    base_parser.add_argument("--radar_duty_ratio", type=float, default=0.01, help="雷达占空比")
    base_parser.add_argument("--var_range_fluctuation", type=float, default=1e-14, help="范围波动过程方差")
    base_parser.add_argument("--radar_impulse_duration", type=float, default=2e-5, help="雷达脉冲持续时间")
    base_parser.add_argument("--radar_spectrum_shape", type=float, default=pi / sqrt(3), help="雷达频谱形状参数")

    # 基于惩罚的 CCCP 算法参数
    base_parser.add_argument("--max_iterations", type=int, default=10, help="CCCP 算法最大迭代次数")
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-6, help="CCCP 算法目标函数收敛阈值")
    base_parser.add_argument("--rank1_threshold", type=float, default=1e-6, help="CCCP 算法秩一约束收敛阈值")
    base_parser.add_argument("--penalty_factor", type=float, default=0.1, help="罚因子")
    base_parser.add_argument("--zoom_factor", type=float, default=2, help="缩放系数")
    base_parser.add_argument("--enable_cccp_diagnostics", default="True", help="启用 CCCP 诊断，检查下一轮线性化后的贴合性和可行性")
    base_parser.add_argument("--diagnostic_violation_tol", type=float, default=1e-7, help="CCCP 诊断时的违约容差")
    base_parser.add_argument("--diagnostic_top_k", type=int, default=5, help="CCCP 诊断时打印的最大违约约束组数量")
    base_parser.add_argument("--constraint_include_groups", type=str, default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32,4.39,4.40,4.44,4.45,auxiliary_t,var", help="启用约束组，逗号分隔")
    base_parser.add_argument("--constraint_exclude_groups", type=str, default="", help="禁用约束组，逗号分隔")

    base_parser.add_argument("--linearization_psi_floor", type=float, default=1e-10, help="CCCP 线性化里 Psi 的数值下界，避免 1/Psi 和 Psi^{-1} 爆大")
    base_parser.add_argument("--enable_first_iter_rank_boost", type=lambda x: str(x).lower() == "true", default=False,
                             help="开关参数")
    base_parser.add_argument("--first_iter_rank_boost_eps", type=float, default=0.1,
                             help="强度参数")
    base_parser.add_argument("--solver_backend", type=str, default="fusion", choices=["fusion", "cvxpy"],
                             help="是否采用 Fusion 求解器，默认为 True（使用 Mosek Fusion），否则使用 CVXPY（默认使用 Mosek 作为 CVXPY 的求解器）")
    base_parser.add_argument("--enable_initial_anchor", type=lambda x: str(x).lower() == "true", default=False,
                             help="是否启用初始化描点/锚点可行性检查，默认为 False")
    return base_parser.parse_args()


def compute_largest_eigenvector(matrix):
    """
    计算矩阵中最大特征值对应的特征向量

    :param matrix: 需要计算最大特征值对应的特征向量的矩阵
    :return: 矩阵中最大特征值对应的特征向量
    """
    # 确保矩阵的对称性，防止数值截断误差
    matrix = (matrix + matrix.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # eigh 返回特征值升序排列，取最后一个
    v_max = eigenvectors[:, -1]
    # 返回最大特征值对应的特征向量
    return v_max


def complex_to_real_lift(matrix):
    matrix = np.asarray(matrix)
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    return np.block([
        [real_part, -imag_part],
        [imag_part, real_part],
    ]).astype(float, copy=False)


def real_lift_to_complex_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    lifted_dim = matrix.shape[0]
    if lifted_dim % 2 != 0:
        raise ValueError(f"lifted matrix dimension must be even, got {lifted_dim}")
    n = lifted_dim // 2
    real_part = matrix[:n, :n]
    imag_part = matrix[n:, :n]
    complex_matrix = real_part + 1j * imag_part
    return (complex_matrix + complex_matrix.conj().T) / 2


def _import_mosek_fusion():
    try:
        import mosek.fusion as mf
        import mosek.fusion.pythonic  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Fusion backend requires the Python package 'mosek'. "
            "Please install the MOSEK Python API in the same environment."
        ) from exc
    return mf


def generate_pos(uavs_num, cus_num, targets_num, center, radius, uav_height):
    """
    根据圆心和半径生成 UAVs、CUs 和 targets 的随机位置

    :param uavs_num: UAVs 的数量
    :param cus_num: CUs 的数量
    :param targets_num: targets 的数量
    :param center: 圆心坐标 (x, y)
    :param radius: 半径
    :param uav_height: UAVs 的高度
    :return: UAVs, CUs 和 targets 的随机位置
    """
    # 生成 UAVs 位置
    r_uav = radius * np.sqrt(np.random.rand(uavs_num))
    theta_uav = np.random.rand(uavs_num) * 2 * np.pi
    uavs_pos = np.zeros((uavs_num, 3))
    uavs_pos[:, 0] = center[0] + r_uav * np.cos(theta_uav)
    uavs_pos[:, 1] = center[1] + r_uav * np.sin(theta_uav)
    uavs_pos[:, 2] = uav_height

    # 生成 CUs 位置 (假设在地面 z = 0)
    r_cu = radius * np.sqrt(np.random.rand(cus_num))
    theta_cu = np.random.rand(cus_num) * 2 * np.pi
    cus_pos = np.zeros((cus_num, 3))
    cus_pos[:, 0] = center[0] + r_cu * np.cos(theta_cu)
    cus_pos[:, 1] = center[1] + r_cu * np.sin(theta_cu)
    # cus_pos[:, 2] = 0  # 默认为 0

    # 生成 targets 位置 (假设在地面 z = 0)
    r_target = radius * np.sqrt(np.random.rand(targets_num))
    theta_target = np.random.rand(targets_num) * 2 * np.pi
    targets_pos = np.zeros((targets_num, 3))
    targets_pos[:, 0] = center[0] + r_target * np.cos(theta_target)
    targets_pos[:, 1] = center[1] + r_target * np.sin(theta_target)
    # targets_pos[:, 2] = 0  # 默认为 0

    return uavs_pos, cus_pos, targets_pos


def compute_com_channel_gain(uavs_pos, cus_pos, ref_path_loss, frac_d_lambda, alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums):
    """
    UAVs, CUs 和 BS 之间的均建模为莱斯路径损耗模型
    根据莱斯路径损耗模型计算 UAVs、CUs 和 BS 之间的通信信道增益

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param cus_pos: CUs 的位置 (J * 3)
    :param ref_path_loss: 1m 路径参考路径损耗
    :param frac_d_lambda: 天线间距与波长的比例
    :param alpha_uav_link: 与 UAV 有关链路的路径损耗系数
    :param alpha_cu_link: 与 CU 有关的路径损耗系数
    :param rician_factor: Rician 因子
    :param antenna_nums: 天线数量 N
    :return: UAVs -> CUs 信道 (I * J * N), UAVs -> BS 信道 (I * 1 * N), CUs -> BS 信道 (J * 1)
    """
    bs_pos = np.array([0, 0, 0])  # 假设 BS 位于原点

    def get_rician_channel(pos1, pos2, alpha, K, is_mimo=True):
        # 计算距离
        # pos1: (N_pos1, 3), pos2: (N_pos2, 3) -> dist: (N_pos1, N_pos2)
        diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # 路径损耗
        path_loss = ref_path_loss * (dist ** -alpha)
        
        if is_mimo:
            # 计算方位角 phi (AoD)
            # diff shape: (N_pos1, N_pos2, 3), diff[..., 0] is dx, diff[..., 1] is dy
            dx = diff[..., 0]
            dy = diff[..., 1]
            phi = np.arctan2(dy, dx)
            
            # LoS 分量 (Array Response Vector)
            # h_los shape: (N_pos1, N_pos2, antenna_nums)
            # array_response: [1, exp(j*2*pi*d/lambda*sin(phi)), ..., exp(j*2*pi*d/lambda*(N-1)*sin(phi))]
            n_range = np.arange(antenna_nums)
            exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(phi)[..., np.newaxis] * n_range
            h_los = np.exp(exponent)
            
            # NLoS 分量 (Rayleigh)
            # h_nlos shape: (N_pos1, N_pos2, antenna_nums)
            h_nlos = (np.random.randn(*dist.shape, antenna_nums) + 1j * np.random.randn(*dist.shape, antenna_nums)) / np.sqrt(2)
            
            # 扩展 path_loss 维度以匹配 MIMO 信道
            path_loss_expanded = path_loss[..., np.newaxis]
            
            h = np.sqrt(path_loss_expanded) * (np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos)
            
        else:
            # SISO 情况 (CU -> BS)
            # LoS 分量 = 1
            h_los = 1.0
            
            # NLoS 分量 (Rayleigh)
            h_nlos = (np.random.randn(*dist.shape) + 1j * np.random.randn(*dist.shape)) / np.sqrt(2)
            
            h = np.sqrt(path_loss) * (np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos)

        return h

    # UAVs -> CUs (MIMO, N antennas at UAV)
    uavs_2_cus_channels = get_rician_channel(uavs_pos, cus_pos, alpha_uav_link, rician_factor, is_mimo=True)
    
    # UAVs -> BS (MIMO, N antennas at UAV)
    uavs_2_bs_channels = get_rician_channel(uavs_pos, bs_pos[np.newaxis, :], alpha_uav_link, rician_factor, is_mimo=True)
    
    # CUs -> BS (SISO, 1 antenna at CU)
    cus_2_bs_channels = get_rician_channel(cus_pos, bs_pos[np.newaxis, :], alpha_cu_link, rician_factor, is_mimo=False)

    return uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels


def compute_sen_channel_gain(radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
    """
    根据雷达截面积、天线间距与波长的比例，UAVs 的位置和 targets 的位置计算 UAVs 与 targets 之间的信道响应矩阵

    :param radar_rcs: 雷达截面积 (xi)
    :param frac_d_lambda: 天线间距与波长的比例
    :param uavs_pos: UAVs 的位置 (I * 3)
    :param targets_pos: targets 的位置 (J * 3)
    :param antenna_nums: 天线数量 N
    :param ref_path_loss: 1m 参考距离下的路径损耗
    :return: UAVs 与 targets 之间的信道响应矩阵 A (I * J * N * N)
    """
    # 计算距离
    # diff: (I, J, 3)
    diff = uavs_pos[:, np.newaxis, :] - targets_pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # 计算方位角 theta (AoD/AoA)
    dx = diff[..., 0]
    dy = diff[..., 1]
    theta = np.arctan2(dy, dx)
    
    # 生成转向矢量 a_t 和 a_r (假设收发共址，角度相同)
    # n_range: (N,)
    n_range = np.arange(antenna_nums)
    # exponent: (I, J, N)
    exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(theta)[..., np.newaxis] * n_range
    # a_vec: (I, J, N)
    a_vec = np.exp(exponent)
    
    # 计算信道矩阵 A = sqrt(xi * ref_path_loss * d^-4) * a_r * a_t^H
    # path_gain_amplitude: (I, J)
    # 注意：原文公式为 sqrt(xi * ref_path_loss * d^-4)，其中 ref_path_loss 应为路径损耗常数
    # 通常雷达方程中接收功率 Pr ~ Pt * G^2 * lambda^2 * RCS / ((4pi)^3 * d^4)
    # 这里按照用户给定的公式实现：sqrt(xi * ref_path_loss * d^-4)
    path_gain_amplitude = np.sqrt(radar_rcs * ref_path_loss * (dist ** -4))
    
    # 扩展维度以进行矩阵乘法
    # a_vec_col: (I, J, N, 1) -> a_r
    a_vec_col = a_vec[..., np.newaxis]
    # a_vec_row_conj: (I, J, 1, N) -> a_t^H
    a_vec_row_conj = np.conj(a_vec)[..., np.newaxis, :]
    
    # matrix_term: (I, J, N, N) = a_r * a_t^H
    matrix_term = np.matmul(a_vec_col, a_vec_row_conj)
    
    # uavs_2_targets_channels: (I, J, N, N)
    uavs_2_targets_channels = path_gain_amplitude[..., np.newaxis, np.newaxis] * matrix_term
    
    return uavs_2_targets_channels


def match_uav_targets_nearest(uavs_pos, targets_pos):
    """
    根据距离最近原则，对 UAVs 和 targets 进行一一匹配
    (使用匈牙利算法最小化总距离)

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param targets_pos: targets 的位置 (J * 3)
    :return: UAVs 和 targets 的匹配矩阵 (I * J)
    """
    num_uavs = uavs_pos.shape[0]
    num_targets = targets_pos.shape[0]
    uavs_targets_matched_matrix = np.zeros((num_uavs, num_targets))
    
    # 计算距离矩阵
    diff = uavs_pos[:, np.newaxis, :] - targets_pos[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # 使用匈牙利算法求解最小权匹配 (距离之和最小)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # 设置匹配矩阵
    uavs_targets_matched_matrix[row_ind, col_ind] = 1
    
    return uavs_targets_matched_matrix


def random_choose_matched_matrix(uavs_pos, cus_pos):
    """
    随机选择 UAVs 和 CUs 的匹配矩阵

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param cus_pos: CUs 的位置 (J * 3)
    :return: UAVs 和 CUs 的匹配矩阵 (I * J)
    """
    num_uavs = uavs_pos.shape[0]
    num_cus = cus_pos.shape[0]
    uavs_cus_matched_matrix = np.zeros((num_uavs, num_cus))
    
    # 生成随机匹配索引
    u_indices = np.arange(num_uavs)
    c_indices = np.arange(num_cus)
    np.random.shuffle(u_indices)
    np.random.shuffle(c_indices)
    
    # 匹配数量取两者的最小值，确保满足 1-to-1 约束
    num_matches = min(num_uavs, num_cus)
    
    for i in range(num_matches):
        # 每个 UAV 索引 u_indices[i] 只会出现一次 -> 每一行最多有一个 1
        # 每个 CU 索引 c_indices[i] 只会出现一次 -> 每一列最多有一个 1
        uavs_cus_matched_matrix[u_indices[i], c_indices[i]] = 1
        
    return uavs_cus_matched_matrix


def extract_matched_sensing_channel(uavs_targets_matched_matrix, uavs_2_targets_channels):
    """
    根据匹配矩阵提取对应的信道响应矩阵

    :param uavs_targets_matched_matrix: UAVs 和 targets 的匹配矩阵 (I * J)
    :param uavs_2_targets_channels: 所有 UAVs 到所有 targets 的信道响应矩阵 (I * J * N * N)
    :return: 匹配后的信道响应矩阵 (I * N * N)
    """
    # 找到匹配的索引
    # row_indices: UAV 索引, col_indices: Target 索引
    row_indices, col_indices = np.where(uavs_targets_matched_matrix == 1)
    
    # 按照 UAV 索引排序 (确保顺序对应 UAV 0, UAV 1, ...)
    sorted_args = np.argsort(row_indices)
    row_indices = row_indices[sorted_args]
    col_indices = col_indices[sorted_args]
    
    # 提取信道
    # matched_uav_sensing_channel: (I, N, N)
    matched_uav_sensing_channel = uavs_2_targets_channels[row_indices, col_indices, :, :]
    
    return matched_uav_sensing_channel


def compute_uav_pos_cur(args, uavs_pos_pre):
    """
    根据前一时刻位置和当前时刻速度计算当前时刻位置

    :param uav_pos_pre: UAVs 前一时刻位置 (I * 3)
    :return: UAVs 当前时刻位置 (I * 3)
    """
    # 根据速度约束生成当前位置
    # 随机生成速度向量，模长在 [min_speed, max_speed] 之间
    # 速度向量 v = [vx, vy, vz], |v| = speed
    # q(t+1) = q(t) + v * tau
    
    # 1. 速度大小
    uav_speed_magnitude = np.random.uniform(args.uav_min_speed, args.uav_max_speed, args.uavs_num)

    # 2. 只生成 xy 方向
    random_direction_xy = np.random.randn(args.uavs_num, 2)
    random_direction_xy /= np.linalg.norm(random_direction_xy, axis=1, keepdims=True)

    # 3. 位移（只在 xy）
    displacement_xy = random_direction_xy * uav_speed_magnitude[:, np.newaxis] * args.time_slot_duration

    # 4. 拼成 3D 位移（z=0）
    displacement = np.zeros_like(uavs_pos_pre)
    displacement[:, :2] = displacement_xy

    # 5. 更新位置（z 不变）
    uavs_pos_cur = uavs_pos_pre + displacement

    return uavs_pos_cur


def initialize_uav_beams(args, matched_uav_sensing_channel, uavs_2_bs_channels):
    hat_uav_sen_beams = []
    hat_uav_off_beams = []
    full_rank_init = (args.uav_max_power / args.antenna_nums) * np.eye(args.antenna_nums, dtype=complex)
    sen_rank1_mix_ratio = 0.002
    off_rank1_mix_ratio = 0.01

    def normalize_direction(vector):
        vector = np.asarray(vector, dtype=complex).reshape(-1)
        norm_val = np.linalg.norm(vector)
        if norm_val <= 1e-12:
            fallback = np.zeros(args.antenna_nums, dtype=complex)
            fallback[0] = 1.0
            return fallback
        return vector / norm_val

    def build_rank1_dominant_init(direction, mix_ratio):
        direction = normalize_direction(direction)
        rank1_init = args.uav_max_power * np.outer(direction, np.conj(direction))
        beam_init = (1.0 - mix_ratio) * rank1_init + mix_ratio * full_rank_init
        return (beam_init + beam_init.conj().T) / 2

    # ==============================
    # 初始化 UAV 感知波束
    # ==============================
    for i in range(args.uavs_num):
        sen_vec = compute_largest_eigenvector(matched_uav_sensing_channel[i])
        hat_uav_sen_beams.append(build_rank1_dominant_init(sen_vec, sen_rank1_mix_ratio))

    # ==============================
    # 初始化 UAV 卸载感知任务波束
    # ==============================
    for i in range(args.uavs_num):
        off_vec = uavs_2_bs_channels[i, 0, :]
        hat_uav_off_beams.append(build_rank1_dominant_init(off_vec, off_rank1_mix_ratio))

    return hat_uav_sen_beams, hat_uav_off_beams


def initialize_uav_beams_random(args, matched_uav_sensing_channel, uavs_2_bs_channels):
    hat_uav_sen_beams = []
    hat_uav_off_beams = []
    full_rank_init = (args.uav_max_power / args.antenna_nums) * np.eye(args.antenna_nums, dtype=complex)
    sen_rank1_mix_ratio = 0.002
    off_rank1_mix_ratio = 0.01

    def sample_random_direction():
        direction = np.random.randn(args.antenna_nums) + 1j * np.random.randn(args.antenna_nums)
        norm_val = np.linalg.norm(direction)
        if norm_val <= 1e-12:
            fallback = np.zeros(args.antenna_nums, dtype=complex)
            fallback[0] = 1.0
            return fallback
        return direction / norm_val

    def build_rank1_dominant_init(direction, mix_ratio):
        rank1_init = args.uav_max_power * np.outer(direction, np.conj(direction))
        beam_init = (1.0 - mix_ratio) * rank1_init + mix_ratio * full_rank_init
        return (beam_init + beam_init.conj().T) / 2

    for _ in range(args.uavs_num):
        hat_uav_sen_beams.append(build_rank1_dominant_init(sample_random_direction(), sen_rank1_mix_ratio))

    for _ in range(args.uavs_num):
        hat_uav_off_beams.append(build_rank1_dominant_init(sample_random_direction(), off_rank1_mix_ratio))

    return hat_uav_sen_beams, hat_uav_off_beams


def update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                          rank1_sen_proj_mats, rank1_off_proj_mats,
                                          cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                          rank1_const_terms):
    cur_rho = float(cur_penalty_factor.value)
    for i in range(args.uavs_num):
        v_w = compute_largest_eigenvector(hat_uav_sen_beams[i].value)
        v_b = compute_largest_eigenvector(hat_uav_off_beams[i].value)
        v_w_matrix = np.outer(v_w, np.conj(v_w))
        v_b_matrix = np.outer(v_b, np.conj(v_b))
        v_w_matrix = (v_w_matrix + v_w_matrix.conj().T) / 2
        v_b_matrix = (v_b_matrix + v_b_matrix.conj().T) / 2
        rank1_sen_proj_mats[i].value = v_w_matrix
        rank1_off_proj_mats[i].value = v_b_matrix
        rank1_sen_scaled_proj_mats[i].value = cur_rho * v_w_matrix
        rank1_off_scaled_proj_mats[i].value = cur_rho * v_b_matrix
        rank1_const_terms.value[i] = float(
            cur_rho * (
                - np.linalg.norm(hat_uav_sen_beams[i].value, 2)
                - np.linalg.norm(hat_uav_off_beams[i].value, 2)
                + np.real(np.trace(v_w_matrix @ hat_uav_sen_beams[i].value))
                + np.real(np.trace(v_b_matrix @ hat_uav_off_beams[i].value))
            )
        )


def update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                         obj5_coef_z, obj5_coef_f, obj5_const_terms):
    alpha = args.omega_weight_1 * args.kappa * args.bs_cycles_per_bit * args.uav_sen_duration
    scale_2 = args.freq_scale ** 2  # 20260321 - 修改了 obj5
    beta = alpha * args.z_scale * scale_2  # 20260321 - 修改了 obj5
    for i in range(args.uavs_num):
        hat_z = float(hat_auxiliary_variable_z.value[i])
        hat_f = float(hat_bs_2_uav_freqs_norm.value[i])
        obj5_coef_z.value[i] = -beta * hat_z  # 20260321 - 修改了 obj5
        obj5_coef_f.value[i] = -2.0 * beta * (hat_f ** 3)  # 20260321 - 修改了 obj5
        obj5_const_terms.value[i] = beta * (0.5 * (hat_z ** 2) + 1.5 * (hat_f ** 4))  # 20260321 - 修改了 obj5

def build_static_constraint_data(args, matched_uav_sensing_channel,
                                 uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels,
                                 uavs_cus_matched_matrix, uavs_off_duration, cus_off_power):
    I = args.uavs_num
    J = args.cus_num
    N = args.antenna_nums
    B = args.bandwidth
    sigma_2 = dbm_2_watt(args.noise_power_density_dbm) * B
    psi_floor = float(args.linearization_psi_floor)
    D_sen = args.uav_sen_duration
    xi1 = args.radar_duty_ratio / (2.0 * args.radar_impulse_duration)
    xi2 = (2.0 * args.var_range_fluctuation
           * (args.radar_spectrum_shape ** 2)
           * (args.bandwidth ** 3)
           * args.radar_impulse_duration)
    eye_n = np.eye(N, dtype=complex)
    eta_matrix = np.asarray(uavs_cus_matched_matrix, dtype=float)
    uavs_off_duration = np.asarray(uavs_off_duration, dtype=float)
    cus_off_power = np.asarray(cus_off_power, dtype=float)
    h_cj_BS_sq = np.asarray([abs(cus_2_bs_channels[j, 0]) ** 2 for j in range(J)], dtype=float)

    Delta_list = []
    AH_Dinv_A_list = []
    logdet_Delta_reg_over_ln2 = np.zeros(I, dtype=float)
    for i in range(I):
        Delta_i = sigma_2 * eye_n.copy()
        for j in range(J):
            eta_ij = eta_matrix[i, j]
            if eta_ij > 0:
                h_ij = uavs_2_cus_channels[i, j, :]
                Delta_i = Delta_i + eta_ij * cus_off_power[j] * np.outer(h_ij, np.conj(h_ij))
        Delta_list.append(Delta_i)

        A_i = matched_uav_sensing_channel[i]
        AH_Dinv_A = A_i.conj().T @ np.linalg.solve(Delta_i, A_i)
        AH_Dinv_A_list.append((AH_Dinv_A + AH_Dinv_A.conj().T) / 2)

        Delta_i_reg = Delta_i + psi_floor * eye_n
        _, logdet_Delta_i_reg = np.linalg.slogdet(Delta_i_reg)
        logdet_Delta_reg_over_ln2[i] = float(logdet_Delta_i_reg / np.log(2))

    H_ui_BS_list = []
    for i in range(I):
        h_i = uavs_2_bs_channels[i, 0, :]
        H_ui_BS_list.append(np.outer(h_i, np.conj(h_i)))

    interf_plus_noise_vec = eta_matrix @ (cus_off_power * h_cj_BS_sq) + sigma_2
    sum_eta_ij_D_sen_vec = D_sen * np.sum(eta_matrix, axis=0)
    sum_eta_ij_D_off_vec = eta_matrix.T @ uavs_off_duration
    log2_cu_snr_vec = np.log2(1.0 + cus_off_power * h_cj_BS_sq / sigma_2)

    return {
        "A_list": matched_uav_sensing_channel,
        "Delta_list": Delta_list,
        "AH_Dinv_A_list": AH_Dinv_A_list,
        "H_ui_BS_list": H_ui_BS_list,
        "h_cj_BS_sq": h_cj_BS_sq,
        "eta_matrix": eta_matrix,
        "interf_plus_noise_vec": np.asarray(interf_plus_noise_vec, dtype=float),
        "sum_eta_ij_D_sen_vec": np.asarray(sum_eta_ij_D_sen_vec, dtype=float),
        "sum_eta_ij_D_off_vec": np.asarray(sum_eta_ij_D_off_vec, dtype=float),
        "log2_cu_snr_vec": np.asarray(log2_cu_snr_vec, dtype=float),
        "logdet_Delta_reg_over_ln2": logdet_Delta_reg_over_ln2,
        "sigma_2": float(sigma_2),
        "psi_floor": psi_floor,
        "xi1": float(xi1),
        "xi2": float(xi2),
        "B": float(B),
        "eye_n": eye_n,
    }


def update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                               static_constraint_data,
                                               c44_const_terms, c44_grad_mats,
                                               c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b):
    I = args.uavs_num
    J = args.cus_num
    B = static_constraint_data["B"]
    sigma_2 = static_constraint_data["sigma_2"]
    psi_floor = static_constraint_data["psi_floor"]
    xi1 = static_constraint_data["xi1"]
    xi2 = static_constraint_data["xi2"]
    eye_n = static_constraint_data["eye_n"]
    A_list = static_constraint_data["A_list"]
    Delta_list = static_constraint_data["Delta_list"]
    H_ui_BS_list = static_constraint_data["H_ui_BS_list"]
    eta_matrix = static_constraint_data["eta_matrix"]
    sum_eta_ij_D_sen_vec = static_constraint_data["sum_eta_ij_D_sen_vec"]
    sum_eta_ij_D_off_vec = static_constraint_data["sum_eta_ij_D_off_vec"]
    logdet_Delta_reg_over_ln2 = static_constraint_data["logdet_Delta_reg_over_ln2"]

    tr_hat_w_bs = np.zeros(I, dtype=float)
    tr_hat_b_bs = np.zeros(I, dtype=float)
    for i in range(I):
        tr_hat_w_bs[i] = float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_sen_beams[i].value)))
        tr_hat_b_bs[i] = float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_off_beams[i].value)))

    for i in range(I):
        A_i = A_list[i]
        Delta_i = Delta_list[i]
        hat_W_i = hat_uav_sen_beams[i].value
        Psi_i_n = Delta_i + xi2 * (A_i @ hat_W_i @ A_i.conj().T)
        Psi_i_n_reg = Psi_i_n + psi_floor * eye_n
        _, logdet_Psi_i_n = np.linalg.slogdet(Psi_i_n_reg)
        Psiinv_A = np.linalg.solve(Psi_i_n_reg, A_i)
        AH_Psiinv_A = A_i.conj().T @ Psiinv_A
        grad_mat = (xi1 * xi2 / np.log(2)) * AH_Psiinv_A
        c44_grad_mats[i].value = (grad_mat + grad_mat.conj().T) / 2
        tr_grad_W = np.real(np.trace(c44_grad_mats[i].value @ hat_W_i))
        c44_const_terms.value[i] = float(
            -xi1 * logdet_Delta_reg_over_ln2[i]
            + xi1 * (logdet_Psi_i_n / np.log(2))
            - tr_grad_W
        )

    tr_hat_w_sum_vec = eta_matrix.T @ tr_hat_w_bs
    tr_hat_b_sum_vec = eta_matrix.T @ tr_hat_b_bs
    Psi_j1_n_safe_vec = np.maximum(sigma_2 + tr_hat_w_sum_vec, psi_floor)
    Psi_j2_n_safe_vec = np.maximum(sigma_2 + tr_hat_b_sum_vec, psi_floor)
    ln2 = np.log(2)

    c45_term4_const.value = sum_eta_ij_D_sen_vec * B * np.log2(Psi_j1_n_safe_vec)
    c45_term5_const.value = sum_eta_ij_D_off_vec * B * np.log2(Psi_j2_n_safe_vec)
    c45_coef_w.value = np.where(
        sum_eta_ij_D_sen_vec > 0,
        B * sum_eta_ij_D_sen_vec / (ln2 * Psi_j1_n_safe_vec),
        0.0,
    )
    c45_coef_b.value = np.where(
        sum_eta_ij_D_off_vec > 0,
        B * sum_eta_ij_D_off_vec / (ln2 * Psi_j2_n_safe_vec),
        0.0,
    )
    c45_const_w.value = -np.asarray(c45_coef_w.value, dtype=float) * tr_hat_w_sum_vec
    c45_const_b.value = -np.asarray(c45_coef_b.value, dtype=float) * tr_hat_b_sum_vec


def compute_obj_fun(args, var_uavs_sen_beam, var_uavs_off_beam, var_bs_2_uav_freqs_norm, var_auxiliary_variable_z, var_cus_off_duration, var_t,
                    hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm, hat_uav_sen_beams, hat_uav_off_beams,
                    rank1_sen_proj_mats, rank1_off_proj_mats, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms,
                    obj5_coef_z, obj5_coef_f, obj5_const_terms, cus_entertaining_task_size,
                    uavs_off_duration, cus_off_power, uavs_pos_pre, uavs_pos_cur, cur_penalty_factor, use_penalty_rank1):

    obj_fun = 0

    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    omega_weight_3 = args.omega_weight_3
    kappa = args.kappa

    cu_max_delay = args.cu_max_delay

    # ==============================
    # UAV 飞行能耗
    # ==============================
    uavs_pos_cur = np.array(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.array(uavs_pos_pre, dtype=float)
    # 1. 按行求范数，得到 shape 为 (args.uavs_num,) 的距离数组
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    # 2. 引入极小值 epsilon 防止悬停时除以 0
    epsilon = 1e-8
    uav_dist_diff = np.maximum(uav_dist_diff, epsilon)
    uav_fly_energy = ( (args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))
    trace_sen_exprs = [cp.real(cp.trace(var_uavs_sen_beam[i])) for i in range(args.uavs_num)]
    trace_off_exprs = [cp.real(cp.trace(var_uavs_off_beam[i])) for i in range(args.uavs_num)]
    trace_sen_vec = cp.hstack(trace_sen_exprs)
    trace_off_vec = cp.hstack(trace_off_exprs)

    # ==============================
    # 第一项
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += omega_weight_1 * kappa * (
            args.bs_cycles_per_bit *
            args.uav_sen_duration *
            (cp.square(var_t[i]) * (args.freq_scale ** 2) * args.z_scale)  # 20260321 - 修改了 obj5
        ) / 2  # 20260321 - 修改了 obj5

    # ==============================
    # 第二项
    # ==============================
    for j in range(args.cus_num):
        obj_fun += omega_weight_1 * kappa * ((args.bs_cycles_per_bit * cus_entertaining_task_size[j]) ** 3) * \
                   cp.power(cu_max_delay - var_cus_off_duration[j], -2)

    # ==============================
    # 第三项
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += omega_weight_2 * (
            args.uav_sen_duration * trace_sen_exprs[i]
            + uavs_off_duration[i] * trace_off_exprs[i]
            + uav_fly_energy[i]
        )

    # ==============================
    # 第四项
    # ==============================
    for j in range(args.cus_num):
        obj_fun += omega_weight_3 * (
            var_cus_off_duration[j] *
            cus_off_power[j]
        )

    # ==============================
    # 第五项 CCCP 线性化
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += (
            obj5_coef_z[i] * var_auxiliary_variable_z[i]
            + obj5_coef_f[i] * var_bs_2_uav_freqs_norm[i]
            + obj5_const_terms[i]
        )
    
    # 采用 rank-1 惩罚项
    if use_penalty_rank1:
        # ==============================
        # 第六项 rank-1 penalty
        # ==============================
        for i in range(args.uavs_num):
            v_w_scaled_matrix = rank1_sen_scaled_proj_mats[i]
            v_b_scaled_matrix = rank1_off_scaled_proj_mats[i]

            rank1_penalty_gap = (
                cur_penalty_factor * (
                    trace_sen_exprs[i]
                    + trace_off_exprs[i]
                )
                - cp.real(cp.trace(v_w_scaled_matrix @ var_uavs_sen_beam[i]))
                - cp.real(cp.trace(v_b_scaled_matrix @ var_uavs_off_beam[i]))
                + rank1_const_terms[i]
            )
            # obj_fun += cp.pos(rank1_penalty_gap)
            obj_fun += rank1_penalty_gap
    
    return obj_fun


def compute_original_obj_fun_value(args, cur_uavs_sen_beams, cur_uavs_off_beams, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z,
                                   cur_cus_off_duration, cus_entertaining_task_size, uavs_off_duration, cus_off_power, uavs_pos_pre,
                                   uavs_pos_cur, cur_penalty_factor, use_penalty_rank1):
    """
    计算原始目标函数值
    """
    obj_fun = 0.0
    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    omega_weight_3 = args.omega_weight_3
    kappa = args.kappa
    cu_max_delay = args.cu_max_delay
    uavs_pos_cur = np.array(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.array(uavs_pos_pre, dtype=float)
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    uav_dist_diff = np.maximum(uav_dist_diff, 1e-8)
    uav_fly_energy = ((args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))
    for i in range(args.uavs_num):
        f_ui = cur_bs_2_uav_freqs_norm[i] * args.freq_scale
        obj_fun += omega_weight_1 * kappa * args.bs_cycles_per_bit * args.uav_sen_duration * (args.z_scale * cur_auxiliary_variable_z[i]) * (f_ui ** 2)  # 20260321 - 修改了 obj5
    for j in range(args.cus_num):
        denom = max(cu_max_delay - cur_cus_off_duration[j], 1e-8)
        obj_fun += omega_weight_1 * kappa * ((args.bs_cycles_per_bit * cus_entertaining_task_size[j]) ** 3) / (denom ** 2)
    for i in range(args.uavs_num):
        obj_fun += omega_weight_2 * (
            args.uav_sen_duration * np.real(np.trace(cur_uavs_sen_beams[i]))
            + uavs_off_duration[i] * np.real(np.trace(cur_uavs_off_beams[i]))
            + uav_fly_energy[i]
        )
    for j in range(args.cus_num):
        obj_fun += omega_weight_3 * cur_cus_off_duration[j] * cus_off_power[j]
    if use_penalty_rank1:
        for i in range(args.uavs_num):
            w_gap = np.real(np.trace(cur_uavs_sen_beams[i])) - np.linalg.norm(cur_uavs_sen_beams[i], 2)
            b_gap = np.real(np.trace(cur_uavs_off_beams[i])) - np.linalg.norm(cur_uavs_off_beams[i], 2)
            # NOTE - 防止因为数值误差导致 gap 为负数
            obj_fun += cur_penalty_factor * (max(float(w_gap), 0.0) + max(float(b_gap), 0.0)) 
    return float(np.real(obj_fun))


def compute_pure_energy_value(args, cur_uavs_sen_beams, cur_uavs_off_beams, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z,
                              cur_cus_off_duration, cus_entertaining_task_size, uavs_off_duration, cus_off_power, uavs_pos_pre,
                              uavs_pos_cur):
    """
    计算纯能量值，即不考虑罚因子的目标函数值。
    """
    return compute_original_obj_fun_value(
        args=args,
        cur_uavs_sen_beams=cur_uavs_sen_beams,
        cur_uavs_off_beams=cur_uavs_off_beams,
        cur_bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
        cur_auxiliary_variable_z=cur_auxiliary_variable_z,
        cur_cus_off_duration=cur_cus_off_duration,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        cur_penalty_factor=0.0,
        use_penalty_rank1=False,
    )


def compute_per_uav_total_energy_list(args, cur_uavs_sen_beams, cur_uavs_off_beams, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z,
                                      uavs_off_duration, uavs_pos_pre, uavs_pos_cur):
    """
    璁＄畻姣忎釜 UAV 鐨勬€昏兘鑰楋紙浠呭寘鍚?UAV 鐩稿叧鐨?4 閮ㄥ垎锛夈€?
    """
    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    kappa = args.kappa
    uavs_pos_cur = np.array(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.array(uavs_pos_pre, dtype=float)
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    uav_dist_diff = np.maximum(uav_dist_diff, 1e-8)
    uav_fly_energy = ((args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))

    per_uav_energy_list = []
    per_uav_sen_power_list = []
    per_uav_off_power_list = []
    per_uav_bs_freq_list = []
    for i in range(args.uavs_num):
        f_ui = cur_bs_2_uav_freqs_norm[i] * args.freq_scale
        sen_power = float(np.real(np.trace(cur_uavs_sen_beams[i])))
        off_power = float(np.real(np.trace(cur_uavs_off_beams[i])))
        per_uav_energy = (
            omega_weight_1 * kappa * args.bs_cycles_per_bit * args.uav_sen_duration
            * (args.z_scale * cur_auxiliary_variable_z[i]) * (f_ui ** 2)
            + omega_weight_2 * args.uav_sen_duration * sen_power
            + omega_weight_2 * uavs_off_duration[i] * off_power
            + omega_weight_2 * uav_fly_energy[i]
        )
        per_uav_energy_list.append(float(np.real(per_uav_energy)))
        per_uav_sen_power_list.append(sen_power)
        per_uav_off_power_list.append(off_power)
        per_uav_bs_freq_list.append(float(f_ui))

    return per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list


def compute_original_obj5_value(args, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z):
    alpha = args.omega_weight_1 * args.kappa * args.bs_cycles_per_bit * args.uav_sen_duration
    z_val = np.asarray(cur_auxiliary_variable_z, dtype=float) * args.z_scale  # 20260321 - 修改了 obj5
    f_val = np.asarray(cur_bs_2_uav_freqs_norm, dtype=float) * args.freq_scale
    return float(alpha * np.sum(z_val * np.square(f_val)))


def compute_surrogate_obj5_value(args, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z, cur_t,
                                 obj5_coef_z, obj5_coef_f, obj5_const_terms):
    alpha = args.omega_weight_1 * args.kappa * args.bs_cycles_per_bit * args.uav_sen_duration
    scale_2 = args.freq_scale ** 2  # 20260321 - 修改了 obj5
    beta = alpha * args.z_scale * scale_2  # 20260321 - 修改了 obj5
    z_val = np.asarray(cur_auxiliary_variable_z, dtype=float)
    f_val = np.asarray(cur_bs_2_uav_freqs_norm, dtype=float)
    t_val = np.asarray(cur_t, dtype=float)
    coef_z = np.asarray(obj5_coef_z.value, dtype=float)
    coef_f = np.asarray(obj5_coef_f.value, dtype=float)
    const_terms = np.asarray(obj5_const_terms.value, dtype=float)
    surrogate_val = np.sum(
        0.5 * beta * np.square(t_val)  # 20260321 - 修改了 obj5
        + coef_z * z_val
        + coef_f * f_val
        + const_terms
    )
    return float(np.real(surrogate_val))


def compute_original_rank1_penalty_value(cur_uavs_sen_beams, cur_uavs_off_beams, cur_penalty_factor):
    rho = float(cur_penalty_factor)
    penalty_val = 0.0
    for sen_beam, off_beam in zip(cur_uavs_sen_beams, cur_uavs_off_beams):
        w_gap = np.real(np.trace(sen_beam)) - np.linalg.norm(sen_beam, 2)
        b_gap = np.real(np.trace(off_beam)) - np.linalg.norm(off_beam, 2)
        penalty_val += rho * (max(float(w_gap), 0.0) + max(float(b_gap), 0.0))
    return float(penalty_val)


def compute_surrogate_rank1_penalty_value(cur_uavs_sen_beams, cur_uavs_off_beams, cur_penalty_factor,
                                          rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms):
    rho = float(cur_penalty_factor)
    penalty_val = 0.0
    for i, (sen_beam, off_beam) in enumerate(zip(cur_uavs_sen_beams, cur_uavs_off_beams)):
        penalty_val += (
            rho * (np.real(np.trace(sen_beam)) + np.real(np.trace(off_beam)))
            - np.real(np.trace(rank1_sen_scaled_proj_mats[i].value @ sen_beam))
            - np.real(np.trace(rank1_off_scaled_proj_mats[i].value @ off_beam))
            + float(rank1_const_terms.value[i])
        )
    return float(np.real(penalty_val))


def define_constraint(args, var_uavs_sen_beam, var_uavs_off_beam, var_bs_2_uav_freqs_norm, var_auxiliary_variable_z, var_cus_off_duration, var_t,
                      first_iter_rank_boost_lb,
                      hat_uav_sen_beams, hat_uav_off_beams, cus_entertaining_task_size, uavs_off_duration, cus_off_power,
                      static_constraint_data,
                      c44_const_terms, c44_grad_mats, c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b):
    """
    定义问题 P6 的所有约束。
    每条约束均附有与论文公式一一对应的逐行注释，方便核对。
    """
    constraints = []

    # ── 基础标量参数 ──────────────────────────────────────────────────────────
    I        = args.uavs_num
    J        = args.cus_num
    N        = args.antenna_nums
    B        = static_constraint_data["B"]
    sigma_2  = static_constraint_data["sigma_2"]
    D_sen    = args.uav_sen_duration                           # D̄_sen
    D_max_ui = args.uav_max_delay                              # D^max_{u_i}
    D_max_cj = args.cu_max_delay                               # D^max_{c_j}
    C_bit    = args.bs_cycles_per_bit                          # C_{u_i} = C_{c_j}
    P_max    = args.uav_max_power                              # P^max_UAV (W)，已是瓦特，无需再转换
    F_max    = args.bs_max_freq                                # F^max
    eps_snr  = db_2_watt(args.sen_sinr)                        # 感知 SINR 门限 ε

    eye_n = static_constraint_data["eye_n"]
    eta_matrix = static_constraint_data["eta_matrix"]
    AH_Dinv_A_list = static_constraint_data["AH_Dinv_A_list"]
    H_ui_BS_list = static_constraint_data["H_ui_BS_list"]
    h_cj_BS_sq = static_constraint_data["h_cj_BS_sq"]
    interf_plus_noise_vec = static_constraint_data["interf_plus_noise_vec"]
    sum_eta_ij_D_sen_vec = static_constraint_data["sum_eta_ij_D_sen_vec"]
    sum_eta_ij_D_off_vec = static_constraint_data["sum_eta_ij_D_off_vec"]
    log2_cu_snr_vec = static_constraint_data["log2_cu_snr_vec"]
    trace_sen_exprs = [cp.real(cp.trace(var_uavs_sen_beam[i])) for i in range(I)]
    trace_off_exprs = [cp.real(cp.trace(var_uavs_off_beam[i])) for i in range(I)]
    uav_bs_sen_gain_exprs = [cp.real(cp.trace(H_ui_BS_list[i] @ var_uavs_sen_beam[i])) for i in range(I)]
    uav_bs_off_gain_exprs = [cp.real(cp.trace(H_ui_BS_list[i] @ var_uavs_off_beam[i])) for i in range(I)]
    uav_bs_sen_gain_vec = cp.hstack(uav_bs_sen_gain_exprs)
    uav_bs_off_gain_vec = cp.hstack(uav_bs_off_gain_exprs)

    # =========================================================================
    # 约束 (4.5)：D̄_sen + D^off_{u_i} - Σ_{j=1}^{J} η_{i,j} D^off_{c_j} ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            D_sen                                               # D̄_sen
            + uavs_off_duration[i]                             # + D^off_{u_i}
            - cp.sum(                                          # - Σ_{j=1}^{J}
                eta_matrix[i, :] *                             #   η_{i,j}
                var_cus_off_duration                           #   · D^off_{c_j}
            )
            <= 0
        )

    # =========================================================================
    # 约束 (4.12)：
    #   D̄_sen · f_{u_i} + D^off_{u_i} · f_{u_i} - D^max_{u_i} · f_{u_i}
    #   + C_{u_i} · D̄_sen · z_i ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            D_sen * var_bs_2_uav_freqs_norm[i] * args.freq_scale                      # D̄_sen · f_{u_i}
            + uavs_off_duration[i] * var_bs_2_uav_freqs_norm[i] * args.freq_scale    # + D^off_{u_i} · f_{u_i}
            - D_max_ui * var_bs_2_uav_freqs_norm[i] * args.freq_scale                 # - D^max_{u_i} · f_{u_i}
            + C_bit * D_sen * args.z_scale * var_auxiliary_variable_z[i]              # + C_{u_i} · D̄_sen · z_i  # 20260321 - 修改了 obj5
            <= 0
        )

    # =========================================================================
    # 约束 (4.23)：W_i(t) ⪰ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(var_uavs_sen_beam[i] - first_iter_rank_boost_lb * eye_n >> 0)

    # =========================================================================
    # 约束 (4.25)：B_i(t) ⪰ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(var_uavs_off_beam[i] - first_iter_rank_boost_lb * eye_n >> 0)

    # =========================================================================
    # 约束 (4.27)：Tr(W_i) - P^max_UAV ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            trace_sen_exprs[i]                                 # Tr(W_i)
            - P_max                                             # - P^max_UAV
            <= 0
        )

    # =========================================================================
    # 约束 (4.28)：ε - Tr(A^H(θ_i) Δ^{-1}_i A(θ_i) W_i) ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        AH_Dinv_A   = AH_Dinv_A_list[i]                        # A^H Δ^{-1} A，shape (N, N)
        constraints.append(
            eps_snr                                             # ε
            - cp.real(cp.trace(                                 # - Tr(A^H Δ^{-1} A · W_i)
                AH_Dinv_A @ var_uavs_sen_beam[i]
            ))
            <= 0
        )

    # =========================================================================
    # 约束 (4.29)：Tr(B_i) - P^max_UAV ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            trace_off_exprs[i]                                 # Tr(B_i)
            - P_max                                             # - P^max_UAV
            <= 0
        )

    # =========================================================================
    # 约束 (4.32)：
    #   D̄_sen · z_i
    #   - D^off_{u_i} · B · log2( Tr(H_{u_i,BS} B_i) + Σ_j η_{i,j} p_j |h_{c_j,BS}|² + σ² )
    #   + D^off_{u_i} · B · log2( Σ_j η_{i,j} p_j |h_{c_j,BS}|² + σ² )
    #   ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        # Σ_j η_{i,j} · p_j · |h_{c_j,BS}|² + σ²（常数）
        interf_plus_noise = interf_plus_noise_vec[i]

        constraints.append(
            D_sen * args.z_scale * var_auxiliary_variable_z[i]  # 20260321 - 修改了 obj5
            - uavs_off_duration[i] * B * cp.log(uav_bs_off_gain_exprs[i] + interf_plus_noise) / np.log(2)
            + uavs_off_duration[i] * B * np.log2(interf_plus_noise)                     
            <= 0
        )

    # =========================================================================
    # 约束 (4.39)：D^off_{c_j} - D^max_{c_j} < 0，∀c_j ∈ C
    #   保证 f_{c_j} = C_{c_j} L_j / (D^max_{c_j} - D^off_{c_j}) 始终为正
    # =========================================================================
    eps_strict = 1e-10
    for j in range(J):
        constraints.append(
            var_cus_off_duration[j]                            # D^off_{c_j}
            - D_max_cj + eps_strict                            # - D^max_{c_j}
            <= 0
        )

    # =========================================================================
    # 约束 (4.40)：
    #   Σ_{u_i} f_{u_i} + Σ_{c_j} C_{c_j} L_j / (D^max_{c_j} - D^off_{c_j}) - F^max ≤ 0
    # =========================================================================
    sum_freq = cp.sum(var_bs_2_uav_freqs_norm) * args.freq_scale                      # Σ_{u_i} f_{u_i}
    for j in range(J):
        sum_freq = (
            sum_freq
            + C_bit * cus_entertaining_task_size[j] * cp.inv_pos(D_max_cj - var_cus_off_duration[j]) 
        )
    constraints.append(sum_freq - F_max <= 0)

    # =========================================================================
    # 约束 (4.44)：对约束 (4.30) 进行 CCCP 线性化（第 n+1 次迭代）
    #
    #   - ξ1 · log2det(Δ_i)
    #   - z_i
    #   + ξ1 · log2det(Ψ^(n)_i)
    #   + ξ1 ξ2 / ln2 · Tr( A^H(θ_i) · (Ψ^(n)_i)^{-1} · A(θ_i) · (W_i - W^(n)_i) )
    #   ≤ 0，∀u_i ∈ U
    #
    #   其中 Ψ^(n)_i = Δ_i + ξ2 · A(θ_i) · W^(n)_i · A^H(θ_i)
    # =========================================================================
    for i in range(I):
        lhs_44 = (
            c44_const_terms[i]
            - args.z_scale * var_auxiliary_variable_z[i]  # 20260321 - 修改了 obj5
            + cp.real(cp.trace(c44_grad_mats[i] @ var_uavs_sen_beam[i]))
        )
        constraints.append(lhs_44 <= 0)

    # =========================================================================
    # 约束 (4.45)：对约束 (4.31) 进行 CCCP 线性化（第 n+1 次迭代）
    #
    #   L_j
    #   - Σ_i η_{i,j} D̄_sen · B · log2( p_j|h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} W_i) + σ² )
    #   - Σ_i η_{i,j} D^off_{u_i} · B · log2( p_j|h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} B_i) + σ² )
    #   - ( D^off_{c_j} - Σ_i η_{i,j} D̄_sen - Σ_i η_{i,j} D^off_{u_i} )
    #     · B · log2( 1 + p_j |h_{c_j,BS}|² / σ² )
    #   + Σ_i η_{i,j} D̄_sen · B · log2( Ψ^(n)_{j,1} )
    #   + Σ_i η_{i,j} D^off_{u_i} · B · log2( Ψ^(n)_{j,2} )
    #   + B · Σ_i η_{i,j} D̄_sen / ( ln2 · Ψ^(n)_{j,1} )
    #     · Σ_i η_{i,j} Tr( H_{u_i,BS} (W_i - W^(n)_i) )
    #   + B · Σ_i η_{i,j} D^off_{u_i} / ( ln2 · Ψ^(n)_{j,2} )
    #     · Σ_i η_{i,j} Tr( H_{u_i,BS} (B_i - B^(n)_i) )
    #   ≤ 0，∀c_j ∈ C
    #
    #   其中 Ψ^(n)_{j,1} = Σ_i η_{i,j} Tr(H_{u_i,BS} W^(n)_i) + σ²
    #        Ψ^(n)_{j,2} = Σ_i η_{i,j} Tr(H_{u_i,BS} B^(n)_i) + σ²
    # =========================================================================
    for j in range(J):
        p_j     = cus_off_power[j]                              # p_j
        h_cj_sq = h_cj_BS_sq[j]                                # |h_{c_j,BS}|²

        # Σ_i η_{i,j} D̄_sen（常数系数）
        sum_eta_ij_D_sen = sum_eta_ij_D_sen_vec[j]
        # Σ_i η_{i,j} D^off_{u_i}（常数系数）
        sum_eta_ij_D_off = sum_eta_ij_D_off_vec[j]

        # log2(1 + p_j |h_{c_j,BS}|² / σ²)（常数）
        log2_cu_snr = log2_cu_snr_vec[j]

        # ── 构建含优化变量的 CVXPY 表达式 ────────────────────────────────────
        # Σ_i η_{i,j} Tr(H_{u_i,BS} W_i)（含优化变量，用于 log 的参数）
        sum_eta_tr_H_W = cp.sum(
            cp.multiply(eta_matrix[:, j], uav_bs_sen_gain_vec)
        )
        for i in range(0):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                sum_eta_tr_H_W = (
                    sum_eta_tr_H_W
                    + eta_ij                                     # η_{i,j}
                    * cp.real(cp.trace(                          # · Tr(H_{u_i,BS} W_i)
                        H_ui_BS_list[i] @ var_uavs_sen_beam[i]
                    ))
                )

        # Σ_i η_{i,j} Tr(H_{u_i,BS} B_i)（含优化变量，用于 log 的参数）
        sum_eta_tr_H_B = cp.sum(
            cp.multiply(eta_matrix[:, j], uav_bs_off_gain_vec)
        )
        for i in range(0):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                sum_eta_tr_H_B = (
                    sum_eta_tr_H_B
                    + eta_ij                                     # η_{i,j}
                    * cp.real(cp.trace(                          # · Tr(H_{u_i,BS} B_i)
                        H_ui_BS_list[i] @ var_uavs_off_beam[i]
                    ))
                )

        # ── 逐项构建约束 (4.39) ───────────────────────────────────────────────

        # 项①：- Σ_i η_{i,j} D̄_sen · B · log2( p_j|h|² + Σ_i η Tr(H W_i) + σ² )
        if sum_eta_ij_D_sen > 0:
            term_1 = (
                - sum_eta_ij_D_sen * B                          # - Σ_i η_{i,j} D̄_sen · B
                * cp.log(                                        # · log(
                    p_j * h_cj_sq                               #     p_j |h_{c_j,BS}|²
                    + sum_eta_tr_H_W                            #   + Σ_i η Tr(H_{u_i,BS} W_i)
                    + sigma_2                                    #   + σ²
                ) / np.log(2)
            )
        else:
            term_1 = 0.0

        # 项②：- Σ_i η_{i,j} D^off_{u_i} · B · log2( p_j|h|² + Σ_i η Tr(H B_i) + σ² )
        if sum_eta_ij_D_off > 0:
            term_2 = (
                - sum_eta_ij_D_off * B                          # - Σ_i η_{i,j} D^off_{u_i} · B
                * cp.log(                                        # · log(
                    p_j * h_cj_sq                               #     p_j |h_{c_j,BS}|²
                    + sum_eta_tr_H_B                            #   + Σ_i η Tr(H_{u_i,BS} B_i)
                    + sigma_2                                    #   + σ²
                ) / np.log(2)
            )
        else:
            term_2 = 0.0

        # 项③：- ( D^off_{c_j} - Σ_i η D̄_sen - Σ_i η D^off_{u_i} ) · B · log2(1 + p_j|h|²/σ²)
        term_3 = (
            - (var_cus_off_duration[j]                          # -( D^off_{c_j}
               - sum_eta_ij_D_sen                               #    - Σ_i η_{i,j} D̄_sen
               - sum_eta_ij_D_off)                              #    - Σ_i η_{i,j} D^off_{u_i} )
            * B * log2_cu_snr                                   # · B · log2(1 + p_j|h|²/σ²)
        )

        term_4 = c45_term4_const[j]
        term_5 = c45_term5_const[j]

        if sum_eta_ij_D_sen > 0:
            term_6 = c45_coef_w[j] * sum_eta_tr_H_W + c45_const_w[j]
        else:
            term_6 = 0.0

        if sum_eta_ij_D_off > 0:
            term_7 = c45_coef_b[j] * sum_eta_tr_H_B + c45_const_b[j]
        else:
            term_7 = 0.0

        constraints.append(
            cus_entertaining_task_size[j]                       # L_j
            + term_1                                            # 项①
            + term_2                                            # 项②
            + term_3                                            # 项③
            + term_4                                            # 项④
            + term_5                                            # 项⑤
            + term_6                                            # 项⑥
            + term_7                                            # 项⑦
            <= 0
        )

    # 追加为了满足 DCP 而引入的变量 t = z_norm + f_norm^{2}  # 20260321 - 修改了 obj5
    # var_t 为 epigraph 上镜图
    # 2026-03-17 修改： var_t 成为归一化变量
    for i in range(args.uavs_num):
        constraints.append(
            var_auxiliary_variable_z[i] + cp.square(var_bs_2_uav_freqs_norm[i]) - var_t[i] <= 0  # 20260321 - 修改了 obj5
        )  # 20260321 - 修改了 obj5
    
    # 定义变量大于 0 
    for i in range(0):
        constraints += [
            var_auxiliary_variable_z[i] >= 0,
            var_bs_2_uav_freqs_norm[i] >= 0,
            var_t[i] >= 0,
        ]

    for j in range(0):
        constraints += [
            var_cus_off_duration[j] >= 0
        ]

    for i in range(0):
        constraints += [
            var_uavs_sen_beam[i] - first_iter_rank_boost_lb * np.eye(N) >> 0,
            var_uavs_off_beam[i] - first_iter_rank_boost_lb * np.eye(N) >> 0,
        ]

    return constraints


def parse_group_list(group_text):
    """
    解析约束组列表字符串，返回约束组名称列表。
    约束组名称列表中的每个元素为一个字符串，代表一个约束组。
    约束组名称列表中的元素顺序与约束组在问题 P6 中的顺序一致。
    """
    if group_text is None:
        return None
    group_text = str(group_text).strip()
    if group_text == "":
        return None
    # 以","进行输入故以逗号进行分割
    return [item.strip() for item in group_text.split(",") if item.strip()]


def build_constraint_map(args, constraints):
    I = args.uavs_num
    J = args.cus_num
    idx = 0
    constraint_map = {}
    constraint_map["4.5"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.12"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.23"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.25"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.27"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.28"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.29"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.32"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.39"] = constraints[idx:idx + J]
    idx += J
    constraint_map["4.40"] = constraints[idx:idx + 1]
    idx += 1
    constraint_map["4.44"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.45"] = constraints[idx:idx + J]
    idx += J
    constraint_map["auxiliary_t"] = constraints[idx:idx + I]
    idx += I
    constraint_map["var"] = constraints[idx:]
    return constraint_map


def select_constraints(constraint_map, include_groups=None, exclude_groups=None):
    all_groups = list(constraint_map.keys())
    unknown_include = [] if include_groups is None else [g for g in include_groups if g not in constraint_map]
    unknown_exclude = [] if exclude_groups is None else [g for g in exclude_groups if g not in constraint_map]
    if len(unknown_include) > 0 or len(unknown_exclude) > 0:
        raise ValueError(f"未知约束组 include={unknown_include}, exclude={unknown_exclude}, 可选组={all_groups}")

    active_groups = all_groups if include_groups is None else [g for g in all_groups if g in include_groups]
    if exclude_groups is not None:
        active_groups = [g for g in active_groups if g not in exclude_groups]

    selected_constraints = []
    for group in active_groups:
        selected_constraints.extend(constraint_map[group])
    return selected_constraints, active_groups


def _safe_constraint_violation_value(constraint):
    try:
        violation = constraint.violation()
    except Exception:
        return float("inf")

    if violation is None:
        return float("inf")

    violation_arr = np.real(np.asarray(violation, dtype=float))
    if violation_arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(violation_arr)):
        return float("inf")
    return float(np.max(violation_arr))


def summarize_constraint_violations(constraint_map, group_names):
    violation_reports = []
    for group_name in group_names:
        group_constraints = constraint_map.get(group_name, [])
        worst_violation = 0.0
        worst_local_idx = 0
        for local_idx, constraint in enumerate(group_constraints, start=1):
            cur_violation = _safe_constraint_violation_value(constraint)
            if cur_violation > worst_violation:
                worst_violation = cur_violation
                worst_local_idx = local_idx
        violation_reports.append({
            "group": group_name,
            "max_violation": worst_violation,
            "worst_local_idx": worst_local_idx,
            "constraint_count": len(group_constraints),
        })

    violation_reports.sort(key=lambda item: item["max_violation"], reverse=True)
    return violation_reports


def run_cccp_locator(args, iter_count, cur_surrogate_opt_val, cur_original_obj_fun_val, problem,
                     constraint_map, active_groups, var_uavs_sen_beam, var_uavs_off_beam,
                     var_auxiliary_variable_z, var_bs_2_uav_freqs_norm, var_t,
                     cur_penalty_factor, use_penalty_rank1,
                     obj5_coef_z, obj5_coef_f, obj5_const_terms,
                     rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms):
    tol = args.diagnostic_violation_tol
    top_k = max(1, int(args.diagnostic_top_k))

    next_surrogate_at_cur_point = float(np.real(problem.objective.value))
    touching_gap = next_surrogate_at_cur_point - float(cur_original_obj_fun_val)
    transition_gap = next_surrogate_at_cur_point - float(cur_surrogate_opt_val)

    cur_uavs_sen_beams = [beam.value for beam in var_uavs_sen_beam]
    cur_uavs_off_beams = [beam.value for beam in var_uavs_off_beam]
    z_val = np.asarray(var_auxiliary_variable_z.value, dtype=float)
    f_val = np.asarray(var_bs_2_uav_freqs_norm.value, dtype=float)
    t_val = np.asarray(var_t.value, dtype=float)
    t_slack = t_val - z_val - np.square(f_val)  # 20260321 - 修改了 obj5
    max_t_slack = float(np.max(t_slack))
    worst_t_idx = int(np.argmax(t_slack)) + 1

    original_obj5_val = compute_original_obj5_value(
        args=args,
        cur_bs_2_uav_freqs_norm=f_val,
        cur_auxiliary_variable_z=z_val,
    )
    surrogate_obj5_val = compute_surrogate_obj5_value(
        args=args,
        cur_bs_2_uav_freqs_norm=f_val,
        cur_auxiliary_variable_z=z_val,
        cur_t=t_val,
        obj5_coef_z=obj5_coef_z,
        obj5_coef_f=obj5_coef_f,
        obj5_const_terms=obj5_const_terms,
    )
    obj5_gap = surrogate_obj5_val - original_obj5_val

    if use_penalty_rank1:
        original_rank1_penalty_val = compute_original_rank1_penalty_value(
            cur_uavs_sen_beams=cur_uavs_sen_beams,
            cur_uavs_off_beams=cur_uavs_off_beams,
            cur_penalty_factor=float(cur_penalty_factor.value),
        )
        surrogate_rank1_penalty_val = compute_surrogate_rank1_penalty_value(
            cur_uavs_sen_beams=cur_uavs_sen_beams,
            cur_uavs_off_beams=cur_uavs_off_beams,
            cur_penalty_factor=float(cur_penalty_factor.value),
            rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
            rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
            rank1_const_terms=rank1_const_terms,
        )
    else:
        original_rank1_penalty_val = 0.0
        surrogate_rank1_penalty_val = 0.0
    rank1_gap = surrogate_rank1_penalty_val - original_rank1_penalty_val
    residual_touching_gap = touching_gap - obj5_gap - rank1_gap

    linearized_groups = [group for group in ("4.44", "4.45") if group in active_groups]
    linearized_reports = summarize_constraint_violations(constraint_map, linearized_groups)
    active_reports = summarize_constraint_violations(constraint_map, active_groups)

    print(f"[CCCP-DIAG] 第 {iter_count} 轮更新线性化后，将当前解代回下一轮子问题:")
    print(f"[CCCP-DIAG]   当前 surrogate optimum = {float(cur_surrogate_opt_val):.12e}")
    print(f"[CCCP-DIAG]   下一轮 surrogate@current = {next_surrogate_at_cur_point:.12e}")
    print(f"[CCCP-DIAG]   touching gap = surrogate@current - original(current) = {touching_gap:.12e}")
    print(f"[CCCP-DIAG]   transition gap = surrogate@current - current surrogate optimum = {transition_gap:.12e}")
    print(f"[CCCP-DIAG]   max auxiliary_t slack = {max_t_slack:.12e} (i = {worst_t_idx})")
    print(f"[CCCP-DIAG]   original obj5 = {original_obj5_val:.12e}")
    print(f"[CCCP-DIAG]   surrogate obj5 = {surrogate_obj5_val:.12e}")
    print(f"[CCCP-DIAG]   obj5 gap = surrogate obj5 - original obj5 = {obj5_gap:.12e}")
    print(f"[CCCP-DIAG]   original rank1 penalty = {original_rank1_penalty_val:.12e}")
    print(f"[CCCP-DIAG]   surrogate rank1 penalty = {surrogate_rank1_penalty_val:.12e}")
    print(f"[CCCP-DIAG]   rank1 penalty gap = surrogate rank1 - original rank1 = {rank1_gap:.12e}")
    print(f"[CCCP-DIAG]   residual touching gap = total - obj5 gap - rank1 gap = {residual_touching_gap:.12e}")

    if linearized_reports:
        worst_linearized = linearized_reports[0]
        print(f"[CCCP-DIAG]   最坏线性化约束组 = {worst_linearized['group']}, max violation = {worst_linearized['max_violation']:.12e}, local idx = {worst_linearized['worst_local_idx']}")
        for report in linearized_reports[:top_k]:
            print(f"[CCCP-DIAG]     group {report['group']}: max violation = {report['max_violation']:.12e}, local idx = {report['worst_local_idx']}")

    flagged_reports = [report for report in active_reports if report["max_violation"] > tol]
    if flagged_reports:
        print(f"[CCCP-DIAG]   max violation > tol ({tol:.1e}) 的约束组:")
        for report in flagged_reports[:top_k]:
            print(f"[CCCP-DIAG]     group {report['group']}: max violation = {report['max_violation']:.12e}, local idx = {report['worst_local_idx']}")
    else:
        print(f"[CCCP-DIAG]   所有激活约束在 tol = {tol:.1e} 内仍保持可行")


def probe_group_feasibility(constraint_map, active_groups, solver):
    """
    问题的可行性检验
    
    :param constraint_map: 约束组 map 集合，key 为约束组名称，value 为约束列表
    :param active_groups: 活动约束组名称列表
    :param solver: 优化求解器
    :return: 每个约束组的可行性检验结果列表，每个元素为 (约束组名称, 求解状态)
    """
    cumulative_constraints = []
    probe_rows = []
    for group in active_groups:
        cumulative_constraints.extend(constraint_map[group])
        probe_problem = cp.Problem(cp.Minimize(0), cumulative_constraints)
        probe_problem.solve(solver=solver, warm_start=False)
        probe_rows.append((group, probe_problem.status))
        if probe_problem.status in ("infeasible", "infeasible_inaccurate"):
            break
    return probe_rows


def resolve_active_constraint_groups(args):
    all_groups = ["4.5", "4.12", "4.23", "4.25", "4.27", "4.28", "4.29", "4.32", "4.39", "4.40", "4.44", "4.45", "auxiliary_t", "var"]
    include_groups = parse_group_list(args.constraint_include_groups)
    exclude_groups = parse_group_list(args.constraint_exclude_groups)
    unknown_include = [] if include_groups is None else [g for g in include_groups if g not in all_groups]
    unknown_exclude = [] if exclude_groups is None else [g for g in exclude_groups if g not in all_groups]
    if len(unknown_include) > 0 or len(unknown_exclude) > 0:
        raise ValueError(f"传参错误:include={unknown_include}, exclude={unknown_exclude}, 可选组={all_groups}")

    active_groups = all_groups if include_groups is None else [g for g in all_groups if g in include_groups]
    if exclude_groups is not None:
        active_groups = [g for g in active_groups if g not in exclude_groups]
    return active_groups


def _get_iteration_schedule(args, fixed_total_iterations):
    if fixed_total_iterations is None:
        return int(args.max_iterations), int(args.max_iterations)
    return 1, int(fixed_total_iterations)


def _fusion_extract_matrix_level(var, dim):
    return np.asarray(var.level(), dtype=float).reshape(dim, dim)


def _build_fusion_failure_result(build_time, solve_time, num_iters, problem_status, solution_status, error_message):
    return {
        "failed": True,
        "status": solution_status,
        "problem_status": problem_status,
        "objective_value": float("inf"),
        "uavs_sen_beams": None,
        "uavs_off_beams": None,
        "bs_2_uav_freqs_norm": None,
        "auxiliary_variable_z": None,
        "cus_off_duration": None,
        "t": None,
        "build_time": float(build_time),
        "solve_time": float(solve_time),
        "num_iters": int(num_iters),
        "error_message": error_message,
    }


def _compute_rank1_gap_metrics(cur_uavs_sen_beams, cur_uavs_off_beams):
    rank1_gap_max = 0.0
    rank1_gap_sum = 0.0
    rank1_sen_gap_sum = 0.0
    rank1_off_gap_sum = 0.0
    for sen_beam, off_beam in zip(cur_uavs_sen_beams, cur_uavs_off_beams):
        w_gap = np.real(np.trace(sen_beam)) - np.linalg.norm(sen_beam, 2)
        b_gap = np.real(np.trace(off_beam)) - np.linalg.norm(off_beam, 2)
        rank1_sen_gap_sum += w_gap
        rank1_off_gap_sum += b_gap
        rank1_gap_sum += w_gap + b_gap
        rank1_gap_max = max(rank1_gap_max, float(max(w_gap, b_gap)))
    return (
        float(rank1_gap_max),
        float(rank1_gap_sum),
        float(rank1_sen_gap_sum),
        float(rank1_off_gap_sum),
    )


def _copy_matrix_list(matrix_list):
    if matrix_list is None:
        return None
    return [None if matrix is None else np.array(matrix, copy=True) for matrix in matrix_list]


def _copy_array(array_value):
    if array_value is None:
        return None
    return np.array(array_value, copy=True)


def _build_penalty_solution_payload(uavs_sen_beams=None,
                                    uavs_off_beams=None,
                                    bs_2_uav_freqs_norm=None,
                                    auxiliary_variable_z=None,
                                    cus_off_duration=None):
    return {
        "uavs_sen_beams": _copy_matrix_list(uavs_sen_beams),
        "uavs_off_beams": _copy_matrix_list(uavs_off_beams),
        "bs_2_uav_freqs_norm": _copy_array(bs_2_uav_freqs_norm),
        "auxiliary_variable_z": _copy_array(auxiliary_variable_z),
        "cus_off_duration": _copy_array(cus_off_duration),
    }


def _is_initial_anchor_enabled(args):
    return bool(getattr(args, "enable_initial_anchor", False))


def _build_fixed_anchor_from_hat_parameters(hat_uav_sen_beams,
                                            hat_uav_off_beams,
                                            hat_bs_2_uav_freqs_norm,
                                            hat_auxiliary_variable_z):
    return {
        "uavs_sen_beams": [np.array(param.value, copy=True) for param in hat_uav_sen_beams],
        "uavs_off_beams": [np.array(param.value, copy=True) for param in hat_uav_off_beams],
        "bs_2_uav_freqs_norm": np.array(hat_bs_2_uav_freqs_norm.value, copy=True),
        "auxiliary_variable_z": np.array(hat_auxiliary_variable_z.value, copy=True),
    }


def _format_penalty_cccp_return(return_values, return_solution=False, solution_payload=None):
    if return_solution:
        return (*return_values, solution_payload)
    return return_values


def _solve_penalty_problem_with_fusion(args, active_groups, static_constraint_data,
                                       hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                       hat_uav_sen_beams, hat_uav_off_beams,
                                       rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms,
                                       obj5_coef_z, obj5_coef_f, obj5_const_terms,
                                       c44_const_terms, c44_grad_mats,
                                       c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b,
                                       cur_penalty_factor, first_iter_rank_boost_lb,
                                       cus_entertaining_task_size, uavs_off_duration, cus_off_power,
                                       uavs_pos_pre, uavs_pos_cur, use_penalty_rank1,
                                       fixed_anchor=None):
    mf = _import_mosek_fusion()
    if "4.23" not in active_groups or "4.25" not in active_groups:
        raise NotImplementedError("Fusion backend currently requires constraint groups 4.23 and 4.25 to remain enabled.")

    build_start = time.perf_counter()
    I = args.uavs_num
    J = args.cus_num
    N = args.antenna_nums
    lifted_dim = 2 * N
    ln2 = np.log(2.0)

    B = static_constraint_data["B"]
    sigma_2 = static_constraint_data["sigma_2"]
    eye_lift = np.eye(lifted_dim, dtype=float)
    eta_matrix = static_constraint_data["eta_matrix"]
    h_cj_BS_sq = static_constraint_data["h_cj_BS_sq"]
    interf_plus_noise_vec = static_constraint_data["interf_plus_noise_vec"]
    sum_eta_ij_D_sen_vec = static_constraint_data["sum_eta_ij_D_sen_vec"]
    sum_eta_ij_D_off_vec = static_constraint_data["sum_eta_ij_D_off_vec"]
    log2_cu_snr_vec = static_constraint_data["log2_cu_snr_vec"]

    H_ui_BS_lift_list = [complex_to_real_lift(mat) for mat in static_constraint_data["H_ui_BS_list"]]
    AH_Dinv_A_lift_list = [complex_to_real_lift(mat) for mat in static_constraint_data["AH_Dinv_A_list"]]
    c44_grad_lift_list = [complex_to_real_lift(param.value) for param in c44_grad_mats]
    rank1_sen_scaled_lift_list = [complex_to_real_lift(param.value) for param in rank1_sen_scaled_proj_mats]
    rank1_off_scaled_lift_list = [complex_to_real_lift(param.value) for param in rank1_off_scaled_proj_mats]
    fixed_sen_lift_list = None
    fixed_off_lift_list = None
    fixed_f_vec = None
    fixed_z_vec = None
    if fixed_anchor is not None:
        fixed_sen_lift_list = [complex_to_real_lift(mat) for mat in fixed_anchor["uavs_sen_beams"]]
        fixed_off_lift_list = [complex_to_real_lift(mat) for mat in fixed_anchor["uavs_off_beams"]]
        fixed_f_vec = np.asarray(fixed_anchor["bs_2_uav_freqs_norm"], dtype=float)
        fixed_z_vec = np.asarray(fixed_anchor["auxiliary_variable_z"], dtype=float)

    D_sen = args.uav_sen_duration
    D_max_ui = args.uav_max_delay
    D_max_cj = args.cu_max_delay
    C_bit = args.bs_cycles_per_bit
    P_max = args.uav_max_power
    F_max = args.bs_max_freq
    eps_snr = db_2_watt(args.sen_sinr)
    eps_strict = 1e-10

    uavs_pos_cur = np.asarray(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.asarray(uavs_pos_pre, dtype=float)
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    uav_dist_diff = np.maximum(uav_dist_diff, 1e-8)
    uav_fly_energy = ((args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))

    obj1_beta = args.omega_weight_1 * args.kappa * args.bs_cycles_per_bit * args.uav_sen_duration * args.z_scale * (args.freq_scale ** 2)

    with mf.Model("penalty_cccp_fusion") as model:
        W_vars = [model.variable(f"W_{i}", mf.Domain.inPSDCone(lifted_dim)) for i in range(I)]
        B_vars = [model.variable(f"B_{i}", mf.Domain.inPSDCone(lifted_dim)) for i in range(I)]
        f_var = model.variable("f", I, mf.Domain.greaterThan(0.0))
        z_var = model.variable("z", I, mf.Domain.greaterThan(0.0))
        d_var = model.variable("d", J, mf.Domain.greaterThan(0.0))
        t_var = model.variable("t", I, mf.Domain.greaterThan(0.0))
        qt_var = model.variable("qt", I, mf.Domain.greaterThan(0.0))
        rinv_var = model.variable("rinv", J, mf.Domain.greaterThan(0.0))
        qinv_var = model.variable("qinv", J, mf.Domain.greaterThan(0.0))
        log_sen_var = model.variable("log_sen", J)
        log_off_var = model.variable("log_off", J)

        trace_sen_exprs = []
        trace_off_exprs = []
        uav_bs_sen_gain_exprs = []
        uav_bs_off_gain_exprs = []
        for i in range(I):
            W_i = W_vars[i]
            B_i = B_vars[i]
            trace_sen_exprs.append(0.5 * mf.Expr.dot(eye_lift, W_i))
            trace_off_exprs.append(0.5 * mf.Expr.dot(eye_lift, B_i))
            uav_bs_sen_gain_exprs.append(0.5 * mf.Expr.dot(H_ui_BS_lift_list[i], W_i))
            uav_bs_off_gain_exprs.append(0.5 * mf.Expr.dot(H_ui_BS_lift_list[i], B_i))

            W11 = W_i.slice([0, 0], [N, N])
            W22 = W_i.slice([N, N], [lifted_dim, lifted_dim])
            W12 = W_i.slice([0, N], [N, lifted_dim])
            W21 = W_i.slice([N, 0], [lifted_dim, N])
            B11 = B_i.slice([0, 0], [N, N])
            B22 = B_i.slice([N, N], [lifted_dim, lifted_dim])
            B12 = B_i.slice([0, N], [N, lifted_dim])
            B21 = B_i.slice([N, 0], [lifted_dim, N])
            model.constraint(W11 - W22, mf.Domain.equalsTo(0.0))
            model.constraint(W12 + W21, mf.Domain.equalsTo(0.0))
            model.constraint(B11 - B22, mf.Domain.equalsTo(0.0))
            model.constraint(B12 + B21, mf.Domain.equalsTo(0.0))

            if "4.23" in active_groups:
                model.constraint(W_i - float(first_iter_rank_boost_lb.value) * eye_lift, mf.Domain.inPSDCone(lifted_dim))
            if "4.25" in active_groups:
                model.constraint(B_i - float(first_iter_rank_boost_lb.value) * eye_lift, mf.Domain.inPSDCone(lifted_dim))
            if fixed_sen_lift_list is not None:
                model.constraint(mf.Expr.sub(W_i, fixed_sen_lift_list[i]), mf.Domain.equalsTo(0.0))
                model.constraint(mf.Expr.sub(B_i, fixed_off_lift_list[i]), mf.Domain.equalsTo(0.0))
                model.constraint(f_var.index(i), mf.Domain.equalsTo(float(fixed_f_vec[i])))
                model.constraint(z_var.index(i), mf.Domain.equalsTo(float(fixed_z_vec[i])))

            model.constraint(mf.Expr.vstack(0.5, qt_var.index(i), t_var.index(i)), mf.Domain.inRotatedQCone())
            if "auxiliary_t" in active_groups:
                model.constraint(mf.Expr.vstack(0.5, t_var.index(i) - z_var.index(i), f_var.index(i)), mf.Domain.inRotatedQCone())

        for j in range(J):
            delay_margin_expr = D_max_cj - d_var.index(j)
            model.constraint(mf.Expr.vstack(delay_margin_expr, rinv_var.index(j), np.sqrt(2.0)), mf.Domain.inRotatedQCone())
            model.constraint(mf.Expr.vstack(0.5, qinv_var.index(j), rinv_var.index(j)), mf.Domain.inRotatedQCone())

        for i in range(I):
            if "4.5" in active_groups:
                model.constraint(D_sen + uavs_off_duration[i] - mf.Expr.dot(eta_matrix[i, :], d_var), mf.Domain.lessThan(0.0))
            if "4.12" in active_groups:
                lhs_412 = (
                    D_sen * args.freq_scale * f_var.index(i)
                    + uavs_off_duration[i] * args.freq_scale * f_var.index(i)
                    - D_max_ui * args.freq_scale * f_var.index(i)
                    + C_bit * D_sen * args.z_scale * z_var.index(i)
                )
                model.constraint(lhs_412, mf.Domain.lessThan(0.0))
            if "4.27" in active_groups:
                model.constraint(trace_sen_exprs[i] - P_max, mf.Domain.lessThan(0.0))
            if "4.28" in active_groups:
                model.constraint(eps_snr - 0.5 * mf.Expr.dot(AH_Dinv_A_lift_list[i], W_vars[i]), mf.Domain.lessThan(0.0))
            if "4.29" in active_groups:
                model.constraint(trace_off_exprs[i] - P_max, mf.Domain.lessThan(0.0))
            if "4.32" in active_groups:
                log_arg_expr = uav_bs_off_gain_exprs[i] + interf_plus_noise_vec[i]
                log_coef = uavs_off_duration[i] * B / ln2
                affine_part = D_sen * args.z_scale * z_var.index(i) + uavs_off_duration[i] * B * np.log2(interf_plus_noise_vec[i])
                model.constraint(mf.Expr.hstack(log_arg_expr, 1.0, affine_part / log_coef), mf.Domain.inPExpCone())
            if "4.44" in active_groups:
                lhs_44 = c44_const_terms.value[i] - args.z_scale * z_var.index(i) + 0.5 * mf.Expr.dot(c44_grad_lift_list[i], W_vars[i])
                model.constraint(lhs_44, mf.Domain.lessThan(0.0))

        if "4.39" in active_groups:
            for j in range(J):
                model.constraint(d_var.index(j) - D_max_cj + eps_strict, mf.Domain.lessThan(0.0))

        if "4.40" in active_groups:
            sum_freq_expr = args.freq_scale * mf.Expr.sum(f_var)
            for j in range(J):
                sum_freq_expr = sum_freq_expr + C_bit * cus_entertaining_task_size[j] * rinv_var.index(j)
            model.constraint(sum_freq_expr - F_max, mf.Domain.lessThan(0.0))

        if "4.45" in active_groups:
            for j in range(J):
                sum_eta_tr_H_W = 0.0
                sum_eta_tr_H_B = 0.0
                for i in range(I):
                    if eta_matrix[i, j] > 0:
                        sum_eta_tr_H_W = sum_eta_tr_H_W + float(eta_matrix[i, j]) * uav_bs_sen_gain_exprs[i]
                        sum_eta_tr_H_B = sum_eta_tr_H_B + float(eta_matrix[i, j]) * uav_bs_off_gain_exprs[i]

                p_j = cus_off_power[j]
                h_cj_sq = h_cj_BS_sq[j]
                sum_eta_ij_D_sen = sum_eta_ij_D_sen_vec[j]
                sum_eta_ij_D_off = sum_eta_ij_D_off_vec[j]
                x1_expr = p_j * h_cj_sq + sum_eta_tr_H_W + sigma_2
                x2_expr = p_j * h_cj_sq + sum_eta_tr_H_B + sigma_2

                if sum_eta_ij_D_sen > 0:
                    model.constraint(mf.Expr.hstack(x1_expr, 1.0, log_sen_var.index(j)), mf.Domain.inPExpCone())
                if sum_eta_ij_D_off > 0:
                    model.constraint(mf.Expr.hstack(x2_expr, 1.0, log_off_var.index(j)), mf.Domain.inPExpCone())

                lhs_45 = (
                    cus_entertaining_task_size[j]
                    - B * log2_cu_snr_vec[j] * d_var.index(j)
                    + (sum_eta_ij_D_sen + sum_eta_ij_D_off) * B * log2_cu_snr_vec[j]
                    + c45_term4_const.value[j]
                    + c45_term5_const.value[j]
                    + c45_const_w.value[j]
                    + c45_const_b.value[j]
                    + c45_coef_w.value[j] * sum_eta_tr_H_W
                    + c45_coef_b.value[j] * sum_eta_tr_H_B
                )
                if sum_eta_ij_D_sen > 0:
                    lhs_45 = lhs_45 - (sum_eta_ij_D_sen * B / ln2) * log_sen_var.index(j)
                if sum_eta_ij_D_off > 0:
                    lhs_45 = lhs_45 - (sum_eta_ij_D_off * B / ln2) * log_off_var.index(j)
                model.constraint(lhs_45, mf.Domain.lessThan(0.0))

        obj_expr = 0.0
        for i in range(I):
            obj_expr = (
                obj_expr
                + 0.5 * obj1_beta * qt_var.index(i)
                + obj5_coef_z.value[i] * z_var.index(i)
                + obj5_coef_f.value[i] * f_var.index(i)
                + obj5_const_terms.value[i]
                + args.omega_weight_2 * (D_sen * trace_sen_exprs[i] + uavs_off_duration[i] * trace_off_exprs[i] + uav_fly_energy[i])
            )
        for j in range(J):
            cu_obj_coef = args.omega_weight_1 * args.kappa * ((args.bs_cycles_per_bit * cus_entertaining_task_size[j]) ** 3)
            obj_expr = obj_expr + cu_obj_coef * qinv_var.index(j) + args.omega_weight_3 * cus_off_power[j] * d_var.index(j)
        if use_penalty_rank1:
            rho = float(cur_penalty_factor.value)
            for i in range(I):
                obj_expr = (
                    obj_expr
                    + rho * (trace_sen_exprs[i] + trace_off_exprs[i])
                    - 0.5 * mf.Expr.dot(rank1_sen_scaled_lift_list[i], W_vars[i])
                    - 0.5 * mf.Expr.dot(rank1_off_scaled_lift_list[i], B_vars[i])
                    + float(rank1_const_terms.value[i])
                )
        model.objective(mf.ObjectiveSense.Minimize, obj_expr)

        build_time = time.perf_counter() - build_start
        solve_start = time.perf_counter()
        model.solve()
        solve_wall_time = time.perf_counter() - solve_start

        try:
            solve_time = float(model.getSolverDoubleInfo("optimizerTime"))
        except Exception:
            solve_time = solve_wall_time
        try:
            num_iters = int(model.getSolverIntInfo("intpntIter"))
        except Exception:
            num_iters = -1

        problem_status = str(model.getProblemStatus())
        solution_status = str(model.getPrimalSolutionStatus())
        problem_status_lower = problem_status.lower()
        solution_status_lower = solution_status.lower()

        if "unknown" in problem_status_lower or "unknown" in solution_status_lower:
            return _build_fusion_failure_result(
                build_time=build_time,
                solve_time=solve_time,
                num_iters=num_iters,
                problem_status=problem_status,
                solution_status=solution_status,
                error_message="Fusion solver did not return an accessible optimal solution.",
            )

        try:
            sen_beams = [real_lift_to_complex_matrix(_fusion_extract_matrix_level(W_vars[i], lifted_dim)) for i in range(I)]
            off_beams = [real_lift_to_complex_matrix(_fusion_extract_matrix_level(B_vars[i], lifted_dim)) for i in range(I)]
            objective_value = float(model.primalObjValue())
            bs_2_uav_freqs_norm = np.asarray(f_var.level(), dtype=float)
            auxiliary_variable_z = np.asarray(z_var.level(), dtype=float)
            cus_off_duration = np.asarray(d_var.level(), dtype=float)
            t_level = np.asarray(t_var.level(), dtype=float)
        except Exception as exc:
            return _build_fusion_failure_result(
                build_time=build_time,
                solve_time=solve_time,
                num_iters=num_iters,
                problem_status=problem_status,
                solution_status=solution_status,
                error_message=str(exc),
            )

        return {
            "failed": False,
            "status": solution_status,
            "problem_status": problem_status,
            "objective_value": objective_value,
            "uavs_sen_beams": sen_beams,
            "uavs_off_beams": off_beams,
            "bs_2_uav_freqs_norm": bs_2_uav_freqs_norm,
            "auxiliary_variable_z": auxiliary_variable_z,
            "cus_off_duration": cus_off_duration,
            "t": t_level,
            "build_time": float(build_time),
            "solve_time": float(solve_time),
            "num_iters": num_iters,
            "error_message": "",
        }


def solve_inner_problem_with_fusion(args, active_groups, static_constraint_data,
                                    hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                    hat_uav_sen_beams, hat_uav_off_beams,
                                    rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms,
                                    obj5_coef_z, obj5_coef_f, obj5_const_terms,
                                    c44_const_terms, c44_grad_mats,
                                    c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b,
                                    cur_penalty_factor, first_iter_rank_boost_lb,
                                    cus_entertaining_task_size, uavs_off_duration, cus_off_power,
                                    uavs_pos_pre, uavs_pos_cur, use_penalty_rank1):
    return _solve_penalty_problem_with_fusion(
        args=args,
        active_groups=active_groups,
        static_constraint_data=static_constraint_data,
        hat_auxiliary_variable_z=hat_auxiliary_variable_z,
        hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
        hat_uav_sen_beams=hat_uav_sen_beams,
        hat_uav_off_beams=hat_uav_off_beams,
        rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
        rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
        rank1_const_terms=rank1_const_terms,
        obj5_coef_z=obj5_coef_z,
        obj5_coef_f=obj5_coef_f,
        obj5_const_terms=obj5_const_terms,
        c44_const_terms=c44_const_terms,
        c44_grad_mats=c44_grad_mats,
        c45_term4_const=c45_term4_const,
        c45_term5_const=c45_term5_const,
        c45_coef_w=c45_coef_w,
        c45_coef_b=c45_coef_b,
        c45_const_w=c45_const_w,
        c45_const_b=c45_const_b,
        cur_penalty_factor=cur_penalty_factor,
        first_iter_rank_boost_lb=first_iter_rank_boost_lb,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        use_penalty_rank1=use_penalty_rank1,
        fixed_anchor=None,
    )


def solve_initial_anchor_problem_with_fusion(args, active_groups, static_constraint_data,
                                             hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                             hat_uav_sen_beams, hat_uav_off_beams,
                                             rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms,
                                             obj5_coef_z, obj5_coef_f, obj5_const_terms,
                                             c44_const_terms, c44_grad_mats,
                                             c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b,
                                             cur_penalty_factor, first_iter_rank_boost_lb,
                                             cus_entertaining_task_size, uavs_off_duration, cus_off_power,
                                             uavs_pos_pre, uavs_pos_cur, use_penalty_rank1):
    fixed_anchor = _build_fixed_anchor_from_hat_parameters(
        hat_uav_sen_beams=hat_uav_sen_beams,
        hat_uav_off_beams=hat_uav_off_beams,
        hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
        hat_auxiliary_variable_z=hat_auxiliary_variable_z,
    )
    return _solve_penalty_problem_with_fusion(
        args=args,
        active_groups=active_groups,
        static_constraint_data=static_constraint_data,
        hat_auxiliary_variable_z=hat_auxiliary_variable_z,
        hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
        hat_uav_sen_beams=hat_uav_sen_beams,
        hat_uav_off_beams=hat_uav_off_beams,
        rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
        rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
        rank1_const_terms=rank1_const_terms,
        obj5_coef_z=obj5_coef_z,
        obj5_coef_f=obj5_coef_f,
        obj5_const_terms=obj5_const_terms,
        c44_const_terms=c44_const_terms,
        c44_grad_mats=c44_grad_mats,
        c45_term4_const=c45_term4_const,
        c45_term5_const=c45_term5_const,
        c45_coef_w=c45_coef_w,
        c45_coef_b=c45_coef_b,
        c45_const_w=c45_const_w,
        c45_const_b=c45_const_b,
        cur_penalty_factor=cur_penalty_factor,
        first_iter_rank_boost_lb=first_iter_rank_boost_lb,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        use_penalty_rank1=use_penalty_rank1,
        fixed_anchor=fixed_anchor,
    )


def penalty_based_cccp_fusion(args,
                              uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                              uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                              uavs_pos_pre, uavs_pos_cur, uavs_off_duration, cus_off_power,
                              use_penalty_rank1=True,
                              cus_entertaining_task_size=None,
                              fixed_total_iterations=None,
                              disable_early_stop=False,
                              return_solution=False):
    """
    基于惩罚的 CCCP 算法求解器，使用 MOSEK Fusion 作为子问题求解器
    """
    if cus_entertaining_task_size is None:
        cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)

    matched_uav_sensing_channel = extract_matched_sensing_channel(
        uavs_targets_matched_matrix=uavs_targets_matched_matrix,
        uavs_2_targets_channels=uavs_2_targets_channels
    )

    hat_auxiliary_variable_z = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * (1e5 / args.z_scale))
    hat_bs_2_uav_freqs_norm = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * (args.bs_max_freq / args.freq_scale / args.uavs_num / 2))
    hat_uav_sen_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    hat_uav_off_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_sen_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_sen_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_z = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_f = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_grad_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums)) for _ in range(args.uavs_num)]
    c45_term4_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_term5_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))

    temp_hat_uav_sen_beams, temp_hat_uav_off_beams = initialize_uav_beams(
        args=args,
        matched_uav_sensing_channel=matched_uav_sensing_channel,
        uavs_2_bs_channels=uavs_2_bs_channels
    )
    for i in range(args.uavs_num):
        hat_uav_sen_beams[i].value = temp_hat_uav_sen_beams[i]
        hat_uav_off_beams[i].value = temp_hat_uav_off_beams[i]

    static_constraint_data = build_static_constraint_data(
        args=args,
        matched_uav_sensing_channel=matched_uav_sensing_channel,
        uavs_2_cus_channels=uavs_2_cus_channels,
        uavs_2_bs_channels=uavs_2_bs_channels,
        cus_2_bs_channels=cus_2_bs_channels,
        uavs_cus_matched_matrix=uavs_cus_matched_matrix,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
    )
    active_groups = resolve_active_constraint_groups(args)

    cur_penalty_factor = cp.Parameter(nonneg=True, value=args.penalty_factor)
    first_iter_rank_boost_lb = cp.Parameter(
        nonneg=True,
        value=args.first_iter_rank_boost_eps if args.enable_first_iter_rank_boost else 0.0
    )

    update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                          rank1_sen_proj_mats, rank1_off_proj_mats,
                                          cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                          rank1_const_terms)
    update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                         obj5_coef_z, obj5_coef_f, obj5_const_terms)
    update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                               static_constraint_data,
                                               c44_const_terms, c44_grad_mats,
                                               c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)

    # print("-------------------------------------------------------- ")
    # print(f"-------------------- CCCP求解开始-------------------- ")
    # print("-------------------------------------------------------- ")
    # print("Fusion backend is enabled. CCCP diagnostics based on CVXPY violation() are skipped.")

    iter_count = 0
    obj_fun_opt = float('inf')
    rank1_gap_max = float('inf')
    original_obj_val_list = []
    energy_val_list = []
    rank1_sen_val_list = []
    rank1_off_val_list = []
    per_uav_energy_list = []
    cur_original_obj_fun_val = float('inf')
    final_solution_payload = None

    outer_iterations, inner_iterations = _get_iteration_schedule(args, fixed_total_iterations)
    if _is_initial_anchor_enabled(args):
        init_problem_result = solve_initial_anchor_problem_with_fusion(
            args=args,
            active_groups=active_groups,
            static_constraint_data=static_constraint_data,
            hat_auxiliary_variable_z=hat_auxiliary_variable_z,
            hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
            hat_uav_sen_beams=hat_uav_sen_beams,
            hat_uav_off_beams=hat_uav_off_beams,
            rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
            rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
            rank1_const_terms=rank1_const_terms,
            obj5_coef_z=obj5_coef_z,
            obj5_coef_f=obj5_coef_f,
            obj5_const_terms=obj5_const_terms,
            c44_const_terms=c44_const_terms,
            c44_grad_mats=c44_grad_mats,
            c45_term4_const=c45_term4_const,
            c45_term5_const=c45_term5_const,
            c45_coef_w=c45_coef_w,
            c45_coef_b=c45_coef_b,
            c45_const_w=c45_const_w,
            c45_const_b=c45_const_b,
            cur_penalty_factor=cur_penalty_factor,
            first_iter_rank_boost_lb=first_iter_rank_boost_lb,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            use_penalty_rank1=use_penalty_rank1,
        )
        if init_problem_result.get("failed", False):
            per_uav_energy_list = [float("inf")] * args.uavs_num; per_uav_sen_power_list = [float("inf")] * args.uavs_num; per_uav_off_power_list = [float("inf")] * args.uavs_num; per_uav_bs_freq_list = [float("inf")] * args.uavs_num; cur_cus_off_duration = [float("inf")] * args.cus_num
            original_obj_val_list.append(float("inf"))
            energy_val_list.append(float("inf"))
            rank1_sen_val_list.append(float("inf"))
            rank1_off_val_list.append(float("inf"))
            return _format_penalty_cccp_return(
                (float("inf"), iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
                return_solution=return_solution,
                solution_payload=final_solution_payload,
            )

        init_surrogate_opt_val = float(np.real(init_problem_result["objective_value"]))
        init_uavs_sen_beams = init_problem_result["uavs_sen_beams"]
        init_uavs_off_beams = init_problem_result["uavs_off_beams"]
        init_bs_2_uav_freqs_norm = init_problem_result["bs_2_uav_freqs_norm"]
        init_auxiliary_variable_z = init_problem_result["auxiliary_variable_z"]
        init_cus_off_duration = init_problem_result["cus_off_duration"]
        cur_cus_off_duration = init_cus_off_duration
        init_original_obj_fun_val = compute_original_obj_fun_value(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            cur_cus_off_duration=init_cus_off_duration,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            cur_penalty_factor=cur_penalty_factor.value,
            use_penalty_rank1=use_penalty_rank1
        )
        init_pure_energy_val = compute_pure_energy_value(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            cur_cus_off_duration=init_cus_off_duration,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur
        )
        per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list = compute_per_uav_total_energy_list(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            uavs_off_duration=uavs_off_duration,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur
        )
        rank1_gap_max, init_rank1_sum, init_rank1_sen_sum, init_rank1_off_sum = _compute_rank1_gap_metrics(init_uavs_sen_beams, init_uavs_off_beams)
        # print(f"第 0 轮 目标函数值 : {init_problem_result['objective_value']}")
        # print(f"第 0 轮 原始目标函数值 {init_original_obj_fun_val}")
        # print(f"第 0 轮 纯能量消耗: {init_pure_energy_val}")
        # print(f"第 0 轮线性化子问题曲线 {init_surrogate_opt_val}")
        # print(f"第 0 轮 最大rank1 gap: {rank1_gap_max}")
        # print(f"第 0 轮 rank1 gap 总和: {init_rank1_sum}")
        # print(f"第 0 轮 w_gap 总和: {init_rank1_sen_sum}")
        # print(f"第 0 轮 b_gap 总和: {init_rank1_off_sum}")
        original_obj_val_list.append(init_original_obj_fun_val)
        energy_val_list.append(init_pure_energy_val)
        rank1_sen_val_list.append(init_rank1_sen_sum)
        rank1_off_val_list.append(init_rank1_off_sum)
        cur_original_obj_fun_val = init_original_obj_fun_val
        obj_fun_opt = init_original_obj_fun_val
        final_solution_payload = _build_penalty_solution_payload(
            uavs_sen_beams=init_uavs_sen_beams,
            uavs_off_beams=init_uavs_off_beams,
            bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            auxiliary_variable_z=init_auxiliary_variable_z,
            cus_off_duration=init_cus_off_duration,
        )

    for outer_iter in range(outer_iterations):
        pre_original_obj_fun_val = float('inf')
        for inner_iter in range(inner_iterations):
            # print(f"-------------------- 第 {iter_count + 1} 次迭代，rho = {cur_penalty_factor.value:.4e} --------------------")
            fusion_result = solve_inner_problem_with_fusion(
                args=args,
                active_groups=active_groups,
                static_constraint_data=static_constraint_data,
                hat_auxiliary_variable_z=hat_auxiliary_variable_z,
                hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
                hat_uav_sen_beams=hat_uav_sen_beams,
                hat_uav_off_beams=hat_uav_off_beams,
                rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
                rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
                rank1_const_terms=rank1_const_terms,
                obj5_coef_z=obj5_coef_z,
                obj5_coef_f=obj5_coef_f,
                obj5_const_terms=obj5_const_terms,
                c44_const_terms=c44_const_terms,
                c44_grad_mats=c44_grad_mats,
                c45_term4_const=c45_term4_const,
                c45_term5_const=c45_term5_const,
                c45_coef_w=c45_coef_w,
                c45_coef_b=c45_coef_b,
                c45_const_w=c45_const_w,
                c45_const_b=c45_const_b,
                cur_penalty_factor=cur_penalty_factor,
                first_iter_rank_boost_lb=first_iter_rank_boost_lb,
                cus_entertaining_task_size=cus_entertaining_task_size,
                uavs_off_duration=uavs_off_duration,
                cus_off_power=cus_off_power,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur,
                use_penalty_rank1=use_penalty_rank1,
            )
            # print("compilation_time =", fusion_result["build_time"])
            # print("solve_time =", fusion_result["solve_time"])
            # print("num_iters =", fusion_result["num_iters"])
            iter_count += 1

            if fusion_result.get("failed", False):
                per_uav_energy_list = [float("inf")] * args.uavs_num; per_uav_sen_power_list = [float("inf")] * args.uavs_num; per_uav_off_power_list = [float("inf")] * args.uavs_num; per_uav_bs_freq_list = [float("inf")] * args.uavs_num; cur_cus_off_duration = [float("inf")] * args.cus_num
                original_obj_val_list.append(float("inf"))
                energy_val_list.append(float("inf"))
                rank1_sen_val_list.append(float("inf"))
                rank1_off_val_list.append(float("inf"))
                # print("内层问题不可解")
                return _format_penalty_cccp_return(
                    (float("inf"), iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
                    return_solution=return_solution,
                    solution_payload=final_solution_payload,
                )

            cur_surrogate_opt_val = float(np.real(fusion_result["objective_value"]))
            cur_uavs_sen_beams = fusion_result["uavs_sen_beams"]
            cur_uavs_off_beams = fusion_result["uavs_off_beams"]
            cur_bs_2_uav_freqs_norm = fusion_result["bs_2_uav_freqs_norm"]
            cur_auxiliary_variable_z = fusion_result["auxiliary_variable_z"]
            cur_cus_off_duration = fusion_result["cus_off_duration"]

            cur_original_obj_fun_val = compute_original_obj_fun_value(
                args=args,
                cur_uavs_sen_beams=cur_uavs_sen_beams,
                cur_uavs_off_beams=cur_uavs_off_beams,
                cur_bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
                cur_auxiliary_variable_z=cur_auxiliary_variable_z,
                cur_cus_off_duration=cur_cus_off_duration,
                cus_entertaining_task_size=cus_entertaining_task_size,
                uavs_off_duration=uavs_off_duration,
                cus_off_power=cus_off_power,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur,
                cur_penalty_factor=cur_penalty_factor.value,
                use_penalty_rank1=use_penalty_rank1
            )
            cur_pure_energy_val = compute_pure_energy_value(
                args=args,
                cur_uavs_sen_beams=cur_uavs_sen_beams,
                cur_uavs_off_beams=cur_uavs_off_beams,
                cur_bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
                cur_auxiliary_variable_z=cur_auxiliary_variable_z,
                cur_cus_off_duration=cur_cus_off_duration,
                cus_entertaining_task_size=cus_entertaining_task_size,
                uavs_off_duration=uavs_off_duration,
                cus_off_power=cus_off_power,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur
            )
            # print(f"第 {iter_count} 轮 目标函数值 : {fusion_result['objective_value']}")
            # print("-------------------------------------------------------- ")
            # print(f"第 {iter_count} 轮 原始目标函数值 {cur_original_obj_fun_val}")
            # print(f"第 {iter_count} 轮 纯能量消耗: {cur_pure_energy_val}")
            # print(f"第 {iter_count} 轮线性化子问题曲线 {cur_surrogate_opt_val}")

            for i in range(args.uavs_num):
                hat_uav_sen_beams[i].value = cur_uavs_sen_beams[i]
                hat_uav_off_beams[i].value = cur_uavs_off_beams[i]
            hat_auxiliary_variable_z.value = cur_auxiliary_variable_z
            hat_bs_2_uav_freqs_norm.value = cur_bs_2_uav_freqs_norm
            per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list = compute_per_uav_total_energy_list(
                args=args,
                cur_uavs_sen_beams=cur_uavs_sen_beams,
                cur_uavs_off_beams=cur_uavs_off_beams,
                cur_bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
                cur_auxiliary_variable_z=cur_auxiliary_variable_z,
                uavs_off_duration=uavs_off_duration,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur
            )

            rank1_gap_max, cur_rank1_sum, cur_rank1_sen_sum, cur_rank1_off_sum = _compute_rank1_gap_metrics(cur_uavs_sen_beams, cur_uavs_off_beams)
            # print(f"第 {iter_count} 轮 最大rank1 gap: {rank1_gap_max}")
            # print(f"第 {iter_count} 轮 rank1 gap 总和: {cur_rank1_sum}")
            # print(f"第 {iter_count} 轮 w_gap 总和: {cur_rank1_sen_sum}")
            # print(f"第 {iter_count} 轮 b_gap 总和: {cur_rank1_off_sum}")
            original_obj_val_list.append(cur_original_obj_fun_val)
            energy_val_list.append(cur_pure_energy_val)
            rank1_sen_val_list.append(cur_rank1_sen_sum)
            rank1_off_val_list.append(cur_rank1_off_sum)
            final_solution_payload = _build_penalty_solution_payload(
                uavs_sen_beams=cur_uavs_sen_beams,
                uavs_off_beams=cur_uavs_off_beams,
                bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
                auxiliary_variable_z=cur_auxiliary_variable_z,
                cus_off_duration=cur_cus_off_duration,
            )

            if args.enable_first_iter_rank_boost and outer_iter == 0 and inner_iter == 0:
                first_iter_rank_boost_lb.value = 0.0

            obj_fun_opt = cur_original_obj_fun_val
            if (not disable_early_stop) and inner_iter != 0 and (abs(cur_original_obj_fun_val - pre_original_obj_fun_val) / abs(pre_original_obj_fun_val)) < args.cccp_threshold:
                # print(f" Convergency!!! - 第 {iter_count} 轮 Fusion求解器求解状态: {fusion_result['status']}")
                # print(f" Convergency!!! - 第 {iter_count} 轮 目标函数值 : {fusion_result['objective_value']}")
                break
            pre_original_obj_fun_val = cur_original_obj_fun_val

            update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                  rank1_sen_proj_mats, rank1_off_proj_mats,
                                                  cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                                  rank1_const_terms)
            update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                                 obj5_coef_z, obj5_coef_f, obj5_const_terms)
            update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                       static_constraint_data,
                                                       c44_const_terms, c44_grad_mats,
                                                       c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)
        if (not disable_early_stop) and cur_rank1_sum < args.rank1_threshold:
            break
        cur_penalty_factor.value *= args.zoom_factor
        update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                              rank1_sen_proj_mats, rank1_off_proj_mats,
                                              cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                              rank1_const_terms)

    return _format_penalty_cccp_return(
        (obj_fun_opt, iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
        return_solution=return_solution,
        solution_payload=final_solution_payload,
    )


def solve_initial_anchor_problem_with_cvxpy(args,
                                            obj_fun,
                                            constraints,
                                            var_uavs_sen_beam,
                                            var_uavs_off_beam,
                                            var_bs_2_uav_freqs_norm,
                                            var_auxiliary_variable_z,
                                            var_cus_off_duration,
                                            var_t,
                                            hat_uav_sen_beams,
                                            hat_uav_off_beams,
                                            hat_bs_2_uav_freqs_norm,
                                            hat_auxiliary_variable_z):
    anchor_constraints = list(constraints)
    for i in range(args.uavs_num):
        anchor_constraints.append(var_uavs_sen_beam[i] == hat_uav_sen_beams[i].value)
        anchor_constraints.append(var_uavs_off_beam[i] == hat_uav_off_beams[i].value)
        anchor_constraints.append(var_bs_2_uav_freqs_norm[i] == hat_bs_2_uav_freqs_norm.value[i])
        anchor_constraints.append(var_auxiliary_variable_z[i] == hat_auxiliary_variable_z.value[i])

    anchor_problem = cp.Problem(cp.Minimize(obj_fun), anchor_constraints)
    anchor_problem.solve(solver=cp.MOSEK, warm_start=True, canon_backend="COO", verbose=False)

    return {
        "status": anchor_problem.status,
        "objective_value": float(np.real(anchor_problem.objective.value)) if anchor_problem.objective.value is not None else float("inf"),
        "uavs_sen_beams": [v.value for v in var_uavs_sen_beam],
        "uavs_off_beams": [v.value for v in var_uavs_off_beam],
        "bs_2_uav_freqs_norm": var_bs_2_uav_freqs_norm.value,
        "auxiliary_variable_z": var_auxiliary_variable_z.value,
        "cus_off_duration": var_cus_off_duration.value,
        "t": var_t.value,
        "compilation_time": anchor_problem.compilation_time,
        "solve_time": anchor_problem.solver_stats.solve_time,
        "num_iters": anchor_problem.solver_stats.num_iters,
    }


def penalty_based_cccp(args,
                       uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                       uavs_pos_pre, uavs_pos_cur, uavs_off_duration, cus_off_power,
                       use_penalty_rank1 = True,
                       cus_entertaining_task_size = None,
                       fixed_total_iterations = None,
                       disable_early_stop = False,
                       return_solution = False
                       ):
    """
    基于惩罚的 CCCP 算法
    
    :param args: 包含所有基础参数的命名空间
    :param uavs_2_cus_channels: UAVs 到 CUs 的信道响应矩阵 (维度: I * J * N * N)
    :param uavs_2_bs_channels: UAVs 到 BS 的信道响应矩阵 (维度: I * 1 * N)
    :param cus_2_bs_channels: CUs 到 BS 的信道响应矩阵 (维度: J * 1)
    :param uavs_2_targets_channels: UAVs 到 targets 的信道响应矩阵 (维度: I * K * N * N)
    :param uavs_targets_matched_matrix: UAVs 和 targets 的匹配矩阵 (维度: I * K) 
    :param uavs_cus_matched_matrix: UAVs 和 CUs 的匹配矩阵 (维度: I * J) (DRL 输出)
    :param uavs_pos_pre: UAVs 前一时刻位置 (维度: I * 3) (DRL 输出)
    :param uavs_pos_cur: UAVs 当前时刻位置 (维度: I * 3) (DRL 输出)
    :param uavs_off_duration: UAVs 感知任务卸载时长 (维度: I) (DRL 输出)
    :param cus_off_power: CUs 娱乐任务卸载功率 (维度: J) (DRL 输出)
    :param use_penalty_rank1: 是否在目标函数中使用基于核范数的 rank-1 惩罚项
    :param cus_entertaining_task_size: CUs 娱乐任务大小 (维度: J)，如果为 None 则在指定范围内随机生成
    :param fixed_total_iterations: 固定的总迭代次数，如果为 None 则使用默认的迭代调度策略
    :param disable_early_stop: 是否禁用基于目标函数值变化率的早停机制
    :param return_solution: 是否额外返回最终波束和辅助变量
    :return: objective_value, total_iterations, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, (optional) solution_payload
    """
    if str(args.solver_backend).lower() == "fusion":
        return penalty_based_cccp_fusion(
            args=args,
            uavs_2_cus_channels=uavs_2_cus_channels,
            uavs_2_bs_channels=uavs_2_bs_channels,
            cus_2_bs_channels=cus_2_bs_channels,
            uavs_2_targets_channels=uavs_2_targets_channels,
            uavs_targets_matched_matrix=uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched_matrix,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            use_penalty_rank1=use_penalty_rank1,
            cus_entertaining_task_size=cus_entertaining_task_size,
            fixed_total_iterations=fixed_total_iterations,
            disable_early_stop=disable_early_stop,
            return_solution=return_solution,
        )

    if cus_entertaining_task_size is None:
        cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)
    
    # 根据匹配矩阵提取对应的信道响应矩阵
    # 结果维度: (I, N, N)，其中 I 是 UAV 数量，N 是天线数量
    matched_uav_sensing_channel \
        = extract_matched_sensing_channel(uavs_targets_matched_matrix = uavs_targets_matched_matrix,
                                          uavs_2_targets_channels = uavs_2_targets_channels
                                          )

    # 最优目标函数值初始化
    obj_fun_opt = float('inf')
    # 声明优化变量
    var_uavs_sen_beam = [cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True) for _ in range(args.uavs_num)]
    var_uavs_off_beam = [cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True) for _ in range(args.uavs_num)]
    var_bs_2_uav_freqs_norm = cp.Variable(args.uavs_num, nonneg=True)
    var_auxiliary_variable_z = cp.Variable(args.uavs_num, nonneg=True)  # 20260321 - 修改了 obj5
    var_cus_off_duration = cp.Variable(args.cus_num, nonneg=True)
    var_t = cp.Variable(args.uavs_num, nonneg=True)  # 辅助变量 t 是为了保障满足 DCP, 该变量没有在文中使用
    # 声明初始值
    hat_auxiliary_variable_z = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * (1e5 / args.z_scale))  # 20260321 - 修改了 obj5
    hat_bs_2_uav_freqs_norm = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * (args.bs_max_freq / args.freq_scale / args.uavs_num / 2))
    hat_uav_sen_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    hat_uav_off_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    # 预计算系数保障 DPP 架构
    rank1_sen_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_sen_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_z = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_f = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_grad_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums)) for _ in range(args.uavs_num)]
    c45_term4_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_term5_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    temp_hat_uav_sen_beams, temp_hat_uav_off_beams = initialize_uav_beams(args = args,
                                                                matched_uav_sensing_channel = matched_uav_sensing_channel,
                                                                uavs_2_bs_channels = uavs_2_bs_channels
                                                                )
    for i in range(args.uavs_num):
        hat_uav_sen_beams[i].value = temp_hat_uav_sen_beams[i]
        hat_uav_off_beams[i].value = temp_hat_uav_off_beams[i]
    static_constraint_data = build_static_constraint_data(
        args=args,
        matched_uav_sensing_channel=matched_uav_sensing_channel,
        uavs_2_cus_channels=uavs_2_cus_channels,
        uavs_2_bs_channels=uavs_2_bs_channels,
        cus_2_bs_channels=cus_2_bs_channels,
        uavs_cus_matched_matrix=uavs_cus_matched_matrix,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
    )
    # 定义罚因子
    cur_penalty_factor = cp.Parameter(nonneg=True, value=args.penalty_factor)
    first_iter_rank_boost_lb = cp.Parameter(
        nonneg=True,
        value=args.first_iter_rank_boost_eps if args.enable_first_iter_rank_boost else 0.0
    )
    # 更新 rank-1 投影矩阵
    update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                          rank1_sen_proj_mats, rank1_off_proj_mats,
                                          cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                          rank1_const_terms)
    update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                         obj5_coef_z, obj5_coef_f, obj5_const_terms)
    update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                               static_constraint_data,
                                               c44_const_terms, c44_grad_mats,
                                               c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)
    
    print("-------------------------------------------------------- ")
    print(f"-------------------- CCCP算法开始迭代 -------------------- ")
    print("-------------------------------------------------------- ")
    iter_count = 0  # 记录迭代次数
    # infeasible_probe_done = False  # 在问题求解为 infeasible 的时候记录是否已经完成了一次不可行性检验
    # 定义目标函数
    obj_fun = compute_obj_fun(args = args,
                            var_uavs_sen_beam = var_uavs_sen_beam,
                            var_uavs_off_beam = var_uavs_off_beam,
                            var_bs_2_uav_freqs_norm = var_bs_2_uav_freqs_norm,
                            var_auxiliary_variable_z = var_auxiliary_variable_z,
                            var_cus_off_duration = var_cus_off_duration,
                            var_t = var_t,
                            hat_auxiliary_variable_z = hat_auxiliary_variable_z,
                            hat_bs_2_uav_freqs_norm = hat_bs_2_uav_freqs_norm,
                            hat_uav_sen_beams = hat_uav_sen_beams,
                            hat_uav_off_beams = hat_uav_off_beams,
                            rank1_sen_proj_mats = rank1_sen_proj_mats,
                            rank1_off_proj_mats = rank1_off_proj_mats,
                            rank1_sen_scaled_proj_mats = rank1_sen_scaled_proj_mats,
                            rank1_off_scaled_proj_mats = rank1_off_scaled_proj_mats,
                            rank1_const_terms = rank1_const_terms,
                            obj5_coef_z = obj5_coef_z,
                            obj5_coef_f = obj5_coef_f,
                            obj5_const_terms = obj5_const_terms,
                            cus_entertaining_task_size = cus_entertaining_task_size,
                            uavs_off_duration = uavs_off_duration,
                            cus_off_power = cus_off_power,
                            uavs_pos_pre = uavs_pos_pre,
                            uavs_pos_cur = uavs_pos_cur,
                            cur_penalty_factor = cur_penalty_factor,
                            use_penalty_rank1 = use_penalty_rank1,
                            )
    all_constraints = define_constraint(args=args,
                                        var_uavs_sen_beam=var_uavs_sen_beam,
                                        var_uavs_off_beam=var_uavs_off_beam,
                                        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
                                        var_auxiliary_variable_z=var_auxiliary_variable_z,
                                        var_cus_off_duration=var_cus_off_duration,
                                        var_t=var_t,
                                        first_iter_rank_boost_lb=first_iter_rank_boost_lb,
                                        hat_uav_sen_beams=hat_uav_sen_beams,
                                        hat_uav_off_beams=hat_uav_off_beams,
                                        cus_entertaining_task_size=cus_entertaining_task_size,
                                        uavs_off_duration=uavs_off_duration,
                                        cus_off_power=cus_off_power,
                                        static_constraint_data=static_constraint_data,
                                        c44_const_terms=c44_const_terms,
                                        c44_grad_mats=c44_grad_mats,
                                        c45_term4_const=c45_term4_const,
                                        c45_term5_const=c45_term5_const,
                                        c45_coef_w=c45_coef_w,
                                        c45_coef_b=c45_coef_b,
                                        c45_const_w=c45_const_w,
                                        c45_const_b=c45_const_b)
    constraint_map = build_constraint_map(args, all_constraints)
    include_groups = parse_group_list(args.constraint_include_groups)
    exclude_groups = parse_group_list(args.constraint_exclude_groups)
    constraints, active_groups = select_constraints(constraint_map,
                                                    include_groups=include_groups,
                                                    exclude_groups=exclude_groups)
    # print(f"启用约束组: {active_groups}")
    # NOTE - 定义为 DPP，加速求解
    problem = cp.Problem(cp.Minimize(obj_fun), constraints)
    print(f"是否定义为 DPP : {problem.is_dcp(dpp=True)}")  # 检测是否为 DPP
    rank1_gap_max = float('inf')
    original_obj_val_list = []
    energy_val_list = []
    rank1_sen_val_list = []
    rank1_off_val_list = []
    per_uav_energy_list = []
    cur_original_obj_fun_val = float('inf')
    final_solution_payload = None
    outer_iterations, inner_iterations = _get_iteration_schedule(args, fixed_total_iterations)
    if _is_initial_anchor_enabled(args):
        init_problem_result = solve_initial_anchor_problem_with_cvxpy(
            args=args,
            obj_fun=obj_fun,
            constraints=constraints,
            var_uavs_sen_beam=var_uavs_sen_beam,
            var_uavs_off_beam=var_uavs_off_beam,
            var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
            var_auxiliary_variable_z=var_auxiliary_variable_z,
            var_cus_off_duration=var_cus_off_duration,
            var_t=var_t,
            hat_uav_sen_beams=hat_uav_sen_beams,
            hat_uav_off_beams=hat_uav_off_beams,
            hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
            hat_auxiliary_variable_z=hat_auxiliary_variable_z,
        )
        if init_problem_result["status"] not in ("optimal", "optimal_inaccurate"):
            print(f"Initial anchor subproblem failed with status: {init_problem_result['status']}")
            original_obj_val_list.append(float("inf"))
            energy_val_list.append(float("inf"))
            rank1_sen_val_list.append(float("inf"))
            rank1_off_val_list.append(float("inf"))
            per_uav_energy_list = [float("inf")] * args.uavs_num; per_uav_sen_power_list = [float("inf")] * args.uavs_num; per_uav_off_power_list = [float("inf")] * args.uavs_num; per_uav_bs_freq_list = [float("inf")] * args.uavs_num; cur_cus_off_duration = [float("inf")] * args.cus_num
            return _format_penalty_cccp_return(
                (float("inf"), iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
                return_solution=return_solution,
                solution_payload=final_solution_payload,
            )

        init_surrogate_opt_val = float(np.real(init_problem_result["objective_value"]))
        init_uavs_sen_beams = init_problem_result["uavs_sen_beams"]
        init_uavs_off_beams = init_problem_result["uavs_off_beams"]
        init_bs_2_uav_freqs_norm = init_problem_result["bs_2_uav_freqs_norm"]
        init_auxiliary_variable_z = init_problem_result["auxiliary_variable_z"]
        init_cus_off_duration = init_problem_result["cus_off_duration"]
        cur_cus_off_duration = init_cus_off_duration
        init_original_obj_fun_val = compute_original_obj_fun_value(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            cur_cus_off_duration=init_cus_off_duration,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            cur_penalty_factor=cur_penalty_factor.value,
            use_penalty_rank1=use_penalty_rank1
        )
        init_pure_energy_val = compute_pure_energy_value(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            cur_cus_off_duration=init_cus_off_duration,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur
        )
        per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list = compute_per_uav_total_energy_list(
            args=args,
            cur_uavs_sen_beams=init_uavs_sen_beams,
            cur_uavs_off_beams=init_uavs_off_beams,
            cur_bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            cur_auxiliary_variable_z=init_auxiliary_variable_z,
            uavs_off_duration=uavs_off_duration,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur
        )
        rank1_gap_max, init_rank1_sum, init_rank1_sen_sum, init_rank1_off_sum = _compute_rank1_gap_metrics(init_uavs_sen_beams, init_uavs_off_beams)
        print(f"第 0 轮 目标函数值 : {init_problem_result['objective_value']}")
        print(f"第 0 轮 原始目标函数值 {init_original_obj_fun_val}")
        print(f"第 0 轮线性化子问题曲线 {init_surrogate_opt_val}")
        print(f"第 0 轮 最大rank1 gap: {rank1_gap_max}")
        print(f"第 0 轮 rank1 gap 总和: {init_rank1_sum}")
        print(f"第 0 轮 w_gap 总和: {init_rank1_sen_sum}")
        print(f"第 0 轮 b_gap 总和: {init_rank1_off_sum}")
        original_obj_val_list.append(init_original_obj_fun_val)
        energy_val_list.append(init_pure_energy_val)
        rank1_sen_val_list.append(init_rank1_sen_sum)
        rank1_off_val_list.append(init_rank1_off_sum)
        cur_original_obj_fun_val = init_original_obj_fun_val
        obj_fun_opt = init_original_obj_fun_val
        final_solution_payload = _build_penalty_solution_payload(
            uavs_sen_beams=init_uavs_sen_beams,
            uavs_off_beams=init_uavs_off_beams,
            bs_2_uav_freqs_norm=init_bs_2_uav_freqs_norm,
            auxiliary_variable_z=init_auxiliary_variable_z,
            cus_off_duration=init_cus_off_duration,
        )

    for outer_iter in range(outer_iterations):
        pre_original_obj_fun_val = float('inf')
        for inner_iter in range(inner_iterations):
            print(f"-------------------- 第 {iter_count + 1} 轮, rho = {cur_penalty_factor.value:.4e} --------------------")
            problem.solve(solver=cp.MOSEK, warm_start=True, canon_backend="COO", verbose=False)
            print("compilation_time =", problem.compilation_time)
            print("solve_time =", problem.solver_stats.solve_time)
            print("num_iters =", problem.solver_stats.num_iters)
            iter_count += 1
            cur_surrogate_opt_val = float(np.real(problem.objective.value))
            if problem.status in ("infeasible", "infeasible_inaccurate"):
                print("当前状态下不存在可行原始解，CVXPY 无法为大多数约束计算 violation。")
                return _format_penalty_cccp_return(
                    (float('inf'), iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
                    return_solution=return_solution,
                    solution_payload=final_solution_payload,
                )
            elif problem.status in ("optimal", "optimal_inaccurate", "unbounded", "unbounded_inaccurate"):
                cur_original_obj_fun_val = compute_original_obj_fun_value(
                    args=args,
                    cur_uavs_sen_beams=[v.value for v in var_uavs_sen_beam],
                    cur_uavs_off_beams=[v.value for v in var_uavs_off_beam],
                    cur_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm.value,
                    cur_auxiliary_variable_z=var_auxiliary_variable_z.value,
                    cur_cus_off_duration=var_cus_off_duration.value,
                    cus_entertaining_task_size=cus_entertaining_task_size,
                    uavs_off_duration=uavs_off_duration,
                    cus_off_power=cus_off_power,
                    uavs_pos_pre=uavs_pos_pre,
                    uavs_pos_cur=uavs_pos_cur,
                    cur_penalty_factor=cur_penalty_factor.value,
                    use_penalty_rank1=use_penalty_rank1
                )
                cur_pure_energy_val = compute_pure_energy_value(
                    args=args,
                    cur_uavs_sen_beams=[v.value for v in var_uavs_sen_beam],
                    cur_uavs_off_beams=[v.value for v in var_uavs_off_beam],
                    cur_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm.value,
                    cur_auxiliary_variable_z=var_auxiliary_variable_z.value,
                    cur_cus_off_duration=var_cus_off_duration.value,
                    cus_entertaining_task_size=cus_entertaining_task_size,
                    uavs_off_duration=uavs_off_duration,
                    cus_off_power=cus_off_power,
                    uavs_pos_pre=uavs_pos_pre,
                    uavs_pos_cur=uavs_pos_cur
                )
                print(f"第 {iter_count} 轮当前状态 : {problem.status}")
                print(f"第 {iter_count} 轮 目标函数值 : {problem.value}")
                print(f"第 {iter_count} 轮 原始目标函数值 {cur_original_obj_fun_val}")
                print(f"第 {iter_count} 轮线性化子问题曲线 {cur_surrogate_opt_val}")
            for i in range(args.uavs_num):
                hat_uav_sen_beams[i].value = var_uavs_sen_beam[i].value
                hat_uav_off_beams[i].value = var_uavs_off_beam[i].value
            hat_auxiliary_variable_z.value = var_auxiliary_variable_z.value
            hat_bs_2_uav_freqs_norm.value = var_bs_2_uav_freqs_norm.value
            per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list = compute_per_uav_total_energy_list(
                args=args,
                cur_uavs_sen_beams=[v.value for v in var_uavs_sen_beam],
                cur_uavs_off_beams=[v.value for v in var_uavs_off_beam],
                cur_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm.value,
                cur_auxiliary_variable_z=var_auxiliary_variable_z.value,
                uavs_off_duration=uavs_off_duration,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur
            )
            rank1_gap_max, cur_rank1_sum, cur_rank1_sen_sum, cur_rank1_off_sum = _compute_rank1_gap_metrics(
                [v.value for v in var_uavs_sen_beam],
                [v.value for v in var_uavs_off_beam]
            )
            print(f"第 {iter_count} 轮最大 rank1 gap:", rank1_gap_max)
            print(f"第 {iter_count} 轮 rank1 gap 总和:", cur_rank1_sum)
            print(f"第 {iter_count} 轮 w_gap 总和:", cur_rank1_sen_sum)
            print(f"第 {iter_count} 轮 b_gap 总和:", cur_rank1_off_sum)
            original_obj_val_list.append(cur_original_obj_fun_val)
            energy_val_list.append(cur_pure_energy_val)
            rank1_sen_val_list.append(cur_rank1_sen_sum)
            rank1_off_val_list.append(cur_rank1_off_sum)
            final_solution_payload = _build_penalty_solution_payload(
                uavs_sen_beams=[v.value for v in var_uavs_sen_beam],
                uavs_off_beams=[v.value for v in var_uavs_off_beam],
                bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm.value,
                auxiliary_variable_z=var_auxiliary_variable_z.value,
                cus_off_duration=var_cus_off_duration.value,
            )

            if args.enable_first_iter_rank_boost and outer_iter == 0 and inner_iter == 0:
                first_iter_rank_boost_lb.value = 0.0

            obj_fun_opt = cur_original_obj_fun_val
            if (not disable_early_stop) and inner_iter != 0 and (abs(cur_original_obj_fun_val - pre_original_obj_fun_val) / abs(pre_original_obj_fun_val)) < args.cccp_threshold:
                print(f" Convergency!!! - 第 {iter_count} 轮求解的问题的 VALUE 值:", cur_original_obj_fun_val)
                print(f" Convergency!!! - 第 {iter_count} 轮rank1 gap 总和:", cur_rank1_sum)
                break
            pre_original_obj_fun_val = cur_original_obj_fun_val
            update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                    rank1_sen_proj_mats, rank1_off_proj_mats,
                                                    cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                                    rank1_const_terms)
            update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                                    obj5_coef_z, obj5_coef_f, obj5_const_terms)
            update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                        static_constraint_data,
                                                        c44_const_terms, c44_grad_mats,
                                                        c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)
            if args.enable_cccp_diagnostics == True:
                run_cccp_locator(
                    args=args,
                    iter_count=iter_count,
                    cur_surrogate_opt_val=cur_surrogate_opt_val,
                    cur_original_obj_fun_val=cur_original_obj_fun_val,
                    problem=problem,
                    constraint_map=constraint_map,
                    active_groups=active_groups,
                    var_uavs_sen_beam=var_uavs_sen_beam,
                    var_uavs_off_beam=var_uavs_off_beam,
                    var_auxiliary_variable_z=var_auxiliary_variable_z,
                    var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
                    var_t=var_t,
                    cur_penalty_factor=cur_penalty_factor,
                    use_penalty_rank1=use_penalty_rank1,
                    obj5_coef_z=obj5_coef_z,
                    obj5_coef_f=obj5_coef_f,
                    obj5_const_terms=obj5_const_terms,
                    rank1_sen_scaled_proj_mats=rank1_sen_scaled_proj_mats,
                    rank1_off_scaled_proj_mats=rank1_off_scaled_proj_mats,
                    rank1_const_terms=rank1_const_terms,
                )
        if (not disable_early_stop) and cur_rank1_sum < args.rank1_threshold:
            break
        cur_penalty_factor.value *= args.zoom_factor
        update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                              rank1_sen_proj_mats, rank1_off_proj_mats,
                                              cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                              rank1_const_terms)

    return _format_penalty_cccp_return(
        (obj_fun_opt, iter_count, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration),
        return_solution=return_solution,
        solution_payload=final_solution_payload,
    )


def penalty_based_cccp_random_init(args,
                                   uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                                   uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                                   uavs_pos_pre, uavs_pos_cur, uavs_off_duration, cus_off_power,
                                   use_penalty_rank1=True,
                                   cus_entertaining_task_size=None,
                                   fixed_total_iterations=None,
                                   disable_early_stop=False):
    original_initializer = globals()["initialize_uav_beams"]
    globals()["initialize_uav_beams"] = initialize_uav_beams_random
    try:
        return penalty_based_cccp(
            args=args,
            uavs_2_cus_channels=uavs_2_cus_channels,
            uavs_2_bs_channels=uavs_2_bs_channels,
            cus_2_bs_channels=cus_2_bs_channels,
            uavs_2_targets_channels=uavs_2_targets_channels,
            uavs_targets_matched_matrix=uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched_matrix,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            use_penalty_rank1=use_penalty_rank1,
            cus_entertaining_task_size=cus_entertaining_task_size,
            fixed_total_iterations=fixed_total_iterations,
            disable_early_stop=disable_early_stop,
        )
    finally:
        globals()["initialize_uav_beams"] = original_initializer


def generate_rho_sweep_energy_csv(args,
                                  uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                                  uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                                  uavs_pos_pre, uavs_pos_cur, uavs_off_duration, cus_off_power,
                                  use_penalty_rank1=True,
                                  cus_entertaining_task_size=None,
                                  rho_values=None,
                                  fixed_total_iterations=20,
                                  csv_prefix=None):
    """
    固定执行 fixed_total_iterations 次正式 CCCP 更新，比较不同 rho 对能耗收敛曲线的影响，并导出 energy CSV。
    若启用初始化描点，则导出的曲线包含额外的初始化锚点求解结果。
    该函数不会触发提前收敛判断，也不会修改传入 args 的原始内容。
    """
    if rho_values is None:
        rho_values = [0.01, 0.1, 1, 10, 100]
    if csv_prefix is None:
        csv_prefix = date_str
    if cus_entertaining_task_size is None:
        cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_paths = []

    for rho in rho_values:
        run_args = argparse.Namespace(**vars(args))
        run_args.penalty_factor = rho
        run_args.enable_cccp_diagnostics = False
        print(f"==================== 正在执行 rho = {rho} 的 CCCP 算法 ====================")
        _, _, _, energy_val_list, _, _, _, _, _, _, _ = penalty_based_cccp(
            args=run_args,
            uavs_2_cus_channels=uavs_2_cus_channels,
            uavs_2_bs_channels=uavs_2_bs_channels,
            cus_2_bs_channels=cus_2_bs_channels,
            uavs_2_targets_channels=uavs_2_targets_channels,
            uavs_targets_matched_matrix=uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched_matrix,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            use_penalty_rank1=use_penalty_rank1,
            cus_entertaining_task_size=cus_entertaining_task_size,
            fixed_total_iterations=fixed_total_iterations,
            disable_early_stop=True,
        )

        output_path = os.path.join(base_dir, f"{csv_prefix}_compare_energy_val_list_rho{rho}.csv")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(energy_val_list)
        output_paths.append(output_path)

    return output_paths


def generate_random_case_convergence_csv(args,
                                         num_cases=None,
                                         csv_prefix=None,
                                         use_penalty_rank1=True):
    """
    随机生成多个有效 case，统计每个 case 的 CCCP 收敛迭代次数，并保存到 CSV 文件。

    若当前 case 求解结果为 INF，则丢弃该 case 并重新随机生成，
    直到收集满 num_cases 个有效 case 为止。
    """
    if num_cases is None:
        num_cases = args.num_cases
    if csv_prefix is None:
        csv_prefix = date_str

    rng = np.random.default_rng(args.seed)
    np_random_state = np.random.get_state()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, f"{csv_prefix}_random_case_convergence_iterations.csv")
    results = []

    try:
        accepted_case_count = 0
        total_attempt_count = 0
        while accepted_case_count < num_cases:
            total_attempt_count += 1
            run_args = argparse.Namespace(**vars(args))
            run_args.uavs_num = int(rng.integers(2, 7))
            run_args.cus_num = int(rng.integers(4, 13))
            run_args.uav_max_delay = float(rng.uniform(0.2, 0.4))
            run_args.enable_cccp_diagnostics = False

            cus_entertaining_task_size = rng.uniform(140e3, 200e3, run_args.cus_num)
            case_seed = int(rng.integers(0, 2 ** 31 - 1))
            np.random.seed(case_seed)

            print("======================================================== ")
            print(f"随机 case 尝试 {total_attempt_count}, 当前已收集有效 case {accepted_case_count}/{num_cases}")
            print(f"UAV 数量 = {run_args.uavs_num}, CU 数量 = {run_args.cus_num}, UAV 最大容忍时延 = {run_args.uav_max_delay:.4f}s")

            uavs_pos, cus_pos, targets_pos = generate_pos(
                uavs_num=run_args.uavs_num,
                cus_num=run_args.cus_num,
                targets_num=run_args.targets_num,
                center=(0, 0),
                radius=run_args.radius,
                uav_height=run_args.uav_height
            )

            uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels = compute_com_channel_gain(
                uavs_pos=uavs_pos,
                cus_pos=cus_pos,
                ref_path_loss=db_2_watt(run_args.ref_path_loss_db),
                frac_d_lambda=run_args.frac_d_lambda,
                alpha_uav_link=run_args.alpha_uav_link,
                alpha_cu_link=run_args.alpha_cu_link,
                rician_factor=db_2_watt(run_args.rician_factor_db),
                antenna_nums=run_args.antenna_nums
            )

            uavs_2_targets_channels = compute_sen_channel_gain(
                radar_rcs=run_args.radar_rcs,
                frac_d_lambda=run_args.frac_d_lambda,
                uavs_pos=uavs_pos,
                targets_pos=targets_pos,
                antenna_nums=run_args.antenna_nums,
                ref_path_loss=db_2_watt(run_args.ref_path_loss_db)
            )

            uavs_cus_matched_matrix = random_choose_matched_matrix(uavs_pos=uavs_pos, cus_pos=cus_pos)
            uavs_targets_matched_matrix = match_uav_targets_nearest(uavs_pos=uavs_pos, targets_pos=targets_pos)
            uavs_pos_cur = compute_uav_pos_cur(args=run_args, uavs_pos_pre=uavs_pos)

            uavs_off_duration = np.full(run_args.uavs_num, (run_args.uav_max_delay - run_args.uav_sen_duration) * 0.8)
            cus_off_power = np.full(run_args.cus_num, dbm_2_watt(run_args.cu_max_power_dbm) / 5)

            objective_value, iter_count, _, _, _, _, _, _, _, _, _ = penalty_based_cccp(
                args=run_args,
                uavs_2_cus_channels=uavs_2_cus_channels,
                uavs_2_bs_channels=uavs_2_bs_channels,
                cus_2_bs_channels=cus_2_bs_channels,
                uavs_2_targets_channels=uavs_2_targets_channels,
                uavs_targets_matched_matrix=uavs_targets_matched_matrix,
                uavs_cus_matched_matrix=uavs_cus_matched_matrix,
                uavs_pos_pre=uavs_pos,
                uavs_pos_cur=uavs_pos_cur,
                uavs_off_duration=uavs_off_duration,
                cus_off_power=cus_off_power,
                use_penalty_rank1=use_penalty_rank1,
                cus_entertaining_task_size=cus_entertaining_task_size,
            )

            if (not np.isfinite(objective_value)) or (not np.isfinite(iter_count)):
                print("当前 case 求解结果为 INF，已丢弃并重新生成新的 case。")
                continue

            accepted_case_count += 1
            results.append({
                "case_id": accepted_case_count,
                "convergence_iterations": iter_count,
            })
    finally:
        np.random.set_state(np_random_state)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "convergence_iterations",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return output_path


def dbm_2_watt(dbm):
    """
    将 dBm 转换为瓦特

    :param dbm: 功率值（dBm）
    :return: 功率值（瓦特）
    """
    return 10 ** ((dbm - 30) / 10)


def db_2_watt(db):
    """
    将 dB 转换为瓦特

    :param db: 功率值（dB）
    :return: 功率值（瓦特）
    """
    return 10 ** (db / 10)


if __name__ == "__main__":
    # 获取参数
    args = get_base_args()
    np.random.seed(args.seed)  # 确保仿真结果可复现
    
    uavs_pos, cus_pos, targets_pos = generate_pos(
        uavs_num=args.uavs_num, cus_num=args.cus_num, targets_num=args.targets_num,
        center=(0, 0), radius=args.radius, uav_height=args.uav_height
    )
    
    uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels = compute_com_channel_gain(
        uavs_pos=uavs_pos, cus_pos=cus_pos, ref_path_loss=db_2_watt(args.ref_path_loss_db),
        frac_d_lambda=args.frac_d_lambda, alpha_uav_link=args.alpha_uav_link,
        alpha_cu_link=args.alpha_cu_link, rician_factor=db_2_watt(args.rician_factor_db),
        antenna_nums=args.antenna_nums
    )
    
    uavs_2_targets_channels = compute_sen_channel_gain(
        radar_rcs=args.radar_rcs, frac_d_lambda=args.frac_d_lambda,
        uavs_pos=uavs_pos, targets_pos=targets_pos,
        antenna_nums=args.antenna_nums, ref_path_loss=db_2_watt(args.ref_path_loss_db)
    )
    
    uavs_cus_matched_matrix = random_choose_matched_matrix(uavs_pos=uavs_pos, cus_pos=cus_pos)
    uavs_targets_matched_matrix = match_uav_targets_nearest(uavs_pos=uavs_pos, targets_pos=targets_pos)
    uavs_pos_cur = compute_uav_pos_cur(args=args, uavs_pos_pre=uavs_pos)
    
    # 固定 DRL 侧输入
    uavs_off_duration = np.full(args.uavs_num, (args.uav_max_delay - args.uav_sen_duration) * 0.8)
    cus_off_power = np.full(args.cus_num, dbm_2_watt(args.cu_max_power_dbm) / 5)
    cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)
    
    
    # # NOTE - 绘图1 - 固定 rho 下的能耗和 rank1 gap 收敛曲线
    # energy_opt, _, original_obj_val_list, energy_val_list, rank1_sen_val_list, rank1_off_val_list, per_uav_energy_list, final_solution = penalty_based_cccp(
    #     args=args,
    #     uavs_2_cus_channels=uavs_2_cus_channels,
    #     uavs_2_bs_channels=uavs_2_bs_channels,
    #     cus_2_bs_channels=cus_2_bs_channels,
    #     uavs_2_targets_channels=uavs_2_targets_channels,
    #     uavs_targets_matched_matrix=uavs_targets_matched_matrix,
    #     uavs_cus_matched_matrix=uavs_cus_matched_matrix,
    #     uavs_pos_pre=uavs_pos,
    #     uavs_pos_cur=uavs_pos_cur,
    #     uavs_off_duration=uavs_off_duration,
    #     cus_off_power=cus_off_power,
    #     use_penalty_rank1=True,
    #     cus_entertaining_task_size=cus_entertaining_task_size,
    #     return_solution=True,
    # )
    # # if final_solution is not None and final_solution["uavs_sen_beams"] is not None:
    # #     beam_pattern_save_path = os.path.join(
    # #         os.path.dirname(os.path.abspath(__file__)),
    # #         f"{date_str}_uav_sensing_beam_patterns_uav{args.uavs_num}.pdf",
    # #     )
    # #     plot_uav_sensing_beam_patterns(
    # #         args=args,
    # #         uavs_sen_beams=final_solution["uavs_sen_beams"],
    # #         uavs_pos=uavs_pos,
    # #         targets_pos=targets_pos,
    # #         uavs_targets_matched_matrix=uavs_targets_matched_matrix,
    # #         save_path=beam_pattern_save_path,
    # #     )
    # with open(f"{date_str}_objective_val_list_rho{args.penalty_factor}_uav{args.uavs_num}.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(original_obj_val_list)
    # with open(f"{date_str}_energy_val_list_rho{args.penalty_factor}_uav{args.uavs_num}.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(energy_val_list)
    # with open(f"{date_str}_rank1_sen_val_list_rho{args.penalty_factor}_uav{args.uavs_num}.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(rank1_sen_val_list)
    # with open(f"{date_str}_rank1_off_val_list_rho{args.penalty_factor}_uav{args.uavs_num}.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(rank1_off_val_list)

    # # NOTE - 绘图2 - 不同 rho 下的能耗收敛曲线
    # generate_rho_sweep_energy_csv(
    #     args=args,
    #     uavs_2_cus_channels=uavs_2_cus_channels,
    #     uavs_2_bs_channels=uavs_2_bs_channels,
    #     cus_2_bs_channels=cus_2_bs_channels,
    #     uavs_2_targets_channels=uavs_2_targets_channels,
    #     uavs_targets_matched_matrix=uavs_targets_matched_matrix,
    #     uavs_cus_matched_matrix=uavs_cus_matched_matrix,
    #     uavs_pos_pre=uavs_pos,
    #     uavs_pos_cur=uavs_pos_cur,
    #     uavs_off_duration=uavs_off_duration,
    #     cus_off_power=cus_off_power,
    #     use_penalty_rank1=True,
    #     cus_entertaining_task_size=cus_entertaining_task_size,
    # )

    # # NOTE - 随机 30 个 case 的收敛次数统计并导出 CSV
    # random_case_csv_path = generate_random_case_convergence_csv(args=args)
    # print(f"随机 case 收敛次数 CSV 已保存到: {random_case_csv_path}")

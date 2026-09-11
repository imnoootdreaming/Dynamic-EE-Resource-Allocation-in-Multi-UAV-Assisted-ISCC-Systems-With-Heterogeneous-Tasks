import argparse
import csv
import os
from datetime import datetime
from math import pi, sqrt
import time
import numpy as np


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def get_base_args():
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=10, help="随机案例数量")
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
    return base_parser.parse_args()


def build_output_csv_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, f"{timestamp}_compare_penalty_gaussian_based_obj_val.csv")


def save_case_results_to_csv(case_results, csv_path):
    fieldnames = [
        "case_id",
        "penalty_based_obj",
        "penalty_based_time",
        "gaussian_based_obj",
        "gaussian_based_time",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_results)


if __name__ == "__main__":
    args = get_base_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    from gaussian_based_cccp_algorithm import gaussian_based_cccp
    from penalty_based_cccp_algorithm import (
        compute_com_channel_gain,
        compute_sen_channel_gain,
        compute_uav_pos_cur,
        db_2_watt,
        dbm_2_watt,
        generate_pos,
        match_uav_targets_nearest,
        penalty_based_cccp,
        random_choose_matched_matrix,
    )

    penalty_based_obj_list = []
    gaussian_based_obj_list = []
    case_results = []

    success_count = 0
    # 使用 while 循环确保最终得到 args.num_cases 个有效案例
    while success_count < args.num_cases:
        current_case_idx = success_count + 1
        print(f"========== Case {current_case_idx}/{args.num_cases} (Attempting...) ==========")

        try:
            # --- 场景参数生成 ---
            args.uavs_num = int(rng.integers(2, 7))
            args.cus_num = int(rng.integers(4, 13))
            args.uav_max_delay = float(rng.uniform(0.2, 0.4))
            cus_entertaining_task_size = rng.uniform(140e3, 200e3, args.cus_num)

            uavs_pos, cus_pos, targets_pos = generate_pos(
                uavs_num=args.uavs_num,
                cus_num=args.cus_num,
                targets_num=args.targets_num,
                center=(0, 0),
                radius=args.radius,
                uav_height=args.uav_height,
            )

            uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels = compute_com_channel_gain(
                uavs_pos=uavs_pos,
                cus_pos=cus_pos,
                ref_path_loss=db_2_watt(args.ref_path_loss_db),
                frac_d_lambda=args.frac_d_lambda,
                alpha_uav_link=args.alpha_uav_link,
                alpha_cu_link=args.alpha_cu_link,
                rician_factor=db_2_watt(args.rician_factor_db),
                antenna_nums=args.antenna_nums,
            )

            uavs_2_targets_channels = compute_sen_channel_gain(
                radar_rcs=args.radar_rcs,
                frac_d_lambda=args.frac_d_lambda,
                uavs_pos=uavs_pos,
                targets_pos=targets_pos,
                antenna_nums=args.antenna_nums,
                ref_path_loss=db_2_watt(args.ref_path_loss_db),
            )

            uavs_cus_matched_matrix = random_choose_matched_matrix(uavs_pos=uavs_pos, cus_pos=cus_pos)
            uavs_targets_matched_matrix = match_uav_targets_nearest(uavs_pos=uavs_pos, targets_pos=targets_pos)
            uavs_pos_cur = compute_uav_pos_cur(args=args, uavs_pos_pre=uavs_pos)

            uavs_off_duration = np.full(args.uavs_num, (args.uav_max_delay - args.uav_sen_duration) * 0.8)
            cus_off_power = np.full(args.cus_num, dbm_2_watt(args.cu_max_power_dbm) / 5)

            # --- 算法 1: Penalty-based ---
            start_penalty = time.time()
            penalty_res = penalty_based_cccp(
                args=args,
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
                use_penalty_rank1=True,
                cus_entertaining_task_size=cus_entertaining_task_size,
            )
            penalty_obj_opt, penalty_iter_count, penalty_energy_history, penalty_rank1_history, _, _, _ = penalty_res
            t_penalty = time.time() - start_penalty
            if penalty_obj_opt == float("inf"):
                continue
            # --- 算法 2: Gaussian-based ---
            start_gaussian = time.time()
            gaussian_res = gaussian_based_cccp(
                args=args,
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
                cus_entertaining_task_size=cus_entertaining_task_size,
                gaussian_randomization_trials = 50
            )
            gaussian_obj_opt, gaussian_iter_count, gaussian_energy_history, gaussian_rank1_history = gaussian_res
            t_gaussian = time.time() - start_gaussian
            if gaussian_obj_opt == float("inf"):
                continue
            # --- 保存结果 ---
            penalty_based_obj_list.append(float(penalty_obj_opt))
            gaussian_based_obj_list.append(float(gaussian_obj_opt))
            case_results.append({
                "case_id": current_case_idx,
                "penalty_based_obj": float(penalty_obj_opt),
                "penalty_based_time": t_penalty,
                "gaussian_based_obj": float(gaussian_obj_opt),
                "gaussian_based_time": t_gaussian,
            })

            print(f"Case {current_case_idx} Success! Penalty Obj: {penalty_obj_opt:.4f}, Gaussian Obj: {gaussian_obj_opt:.4f}")
            success_count += 1  # 只有成功运行到这里，计数器才增加

        except Exception as e:
            print(f"!!! Case {current_case_idx} Failed due to error: {e}. Re-rolling parameters...")
            continue

    # --- 最终输出 ---
    output_csv_path = build_output_csv_path()
    save_case_results_to_csv(case_results=case_results, csv_path=output_csv_path)

    print("========================================================")
    print(f"Comparison finished. Total successful cases: {len(case_results)}")
    print(f"Case result CSV saved to: {output_csv_path}")

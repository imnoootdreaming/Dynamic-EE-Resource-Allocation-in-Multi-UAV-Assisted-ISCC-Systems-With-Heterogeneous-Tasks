import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 内层 PC3P 由同目录桥接模块接入：它把外层实时信道/位置/动作映射为内层 P5 求解上下文。
# 函数名沿用旧接口 penalty_based_cccp，故下方调用点无需改动。
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cccp_bridge import solve_inner_energy as penalty_based_cccp
import numpy as np


class MyReward:
    def __init__(self, base_args):
        self.base_args = base_args
        self.x_max = float(getattr(base_args, "radius", 600))
        self.y_max = float(getattr(base_args, "radius", 600))

    def reward_compute(self, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                       uavs_pos, uavs_pos_cur, uavs_off_duration, cus_off_power, cus_entertaining_task_size,
                       uavs_rec_beam_vectors=None):
        uav_collision_penalty = np.zeros(self.base_args.uavs_num)
        uav_exceed_boundary_penalty = np.zeros(self.base_args.uavs_num)

        for i in range(self.base_args.uavs_num):
            x, y, _ = uavs_pos_cur[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
                uavs_pos_cur[i][0] = np.clip(x, -self.x_max, self.x_max)
                uavs_pos_cur[i][1] = np.clip(y, -self.y_max, self.y_max)

            for j in range(i + 1, self.base_args.uavs_num):
                dist = np.linalg.norm(uavs_pos_cur[i] - uavs_pos_cur[j])
                if dist < self.base_args.uav_safe_distance:
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1

        bs_alloc_spectrum_penalty = np.sum(np.maximum(0, np.sum(uavs_cus_matched_matrix, axis=0) - 1))
        uavs_off_duration = [float(x) for x in np.asarray(uavs_off_duration).reshape(-1)]
        cus_off_power = [float(x) for x in np.asarray(cus_off_power).reshape(-1)]

        energy_opt, _, _, _, _, _, _, per_uav_sen_power_list, per_uav_off_power_list, per_uav_bs_freq_list, cur_cus_off_duration, solution_payload = penalty_based_cccp(
            args=self.base_args,
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
            uavs_rec_beam_vectors=uavs_rec_beam_vectors,
            return_solution=True
        )

        avg_cu_off_power = float(np.mean(cus_off_power)) if len(cus_off_power) > 0 else float("nan")
        bs_cu_freq_list = []
        if len(cur_cus_off_duration) > 0 and np.isfinite(cur_cus_off_duration[0]):
            for j in range(self.base_args.cus_num):
                denom = max(self.base_args.cu_max_delay - float(cur_cus_off_duration[j]), 1e-8)
                f_cu = self.base_args.bs_cycles_per_bit * cus_entertaining_task_size[j] / denom
                bs_cu_freq_list.append(float(f_cu))
        avg_bs_cu_freq = float(np.mean(bs_cu_freq_list)) if bs_cu_freq_list else float("nan")

        # ── 4 个能耗分解值（不加权，口径与 unified_energy_breakdown_test.py 一致）──
        if energy_opt == float("inf"):
            flight_energy = float("nan")
            sensing_energy = float("nan")
            offloading_energy = float("nan")
            computation_energy = float("nan")
        else:
            uav_dist_diff = np.maximum(np.linalg.norm(uavs_pos_cur - uavs_pos, axis=1), 1e-8)
            flight_energy = float(np.sum(
                self.base_args.uav_c1 * (uav_dist_diff ** 3) / (self.base_args.time_slot_duration ** 2)
                + self.base_args.uav_c2 * (self.base_args.time_slot_duration ** 2) / uav_dist_diff
            ))
            sensing_energy = float(self.base_args.uav_sen_duration * np.sum(per_uav_sen_power_list))
            offloading_energy = float(np.sum(np.asarray(uavs_off_duration) * np.asarray(per_uav_off_power_list)))
            if solution_payload is not None and solution_payload.get("auxiliary_variable_z") is not None:
                aux_z = np.asarray(solution_payload["auxiliary_variable_z"], dtype=float).reshape(-1)
            else:
                aux_z = np.zeros(self.base_args.uavs_num, dtype=float)
            f_actual = np.asarray(per_uav_bs_freq_list, dtype=float).reshape(-1)
            computation_energy = float(np.sum(
                self.base_args.kappa * self.base_args.bs_cycles_per_bit * self.base_args.uav_sen_duration
                * self.base_args.z_scale * aux_z * (f_actual ** 2)
            ))

        no_solution_penalty = 0
        if energy_opt == float("inf"):
            no_solution_penalty = 1

        total_reward_4_energy = np.exp(-energy_opt / 1000)
        total_reward = (
            total_reward_4_energy
            - bs_alloc_spectrum_penalty
            - np.sum(uav_exceed_boundary_penalty)
            - np.sum(uav_collision_penalty)
            - no_solution_penalty
        )

        reward = {
            "bs": float(total_reward),
            "components": {
                "total_reward_4_energy": float(total_reward_4_energy),
                "bs_alloc_spectrum_penalty": float(bs_alloc_spectrum_penalty),
                "uav_exceed_boundary_penalty_sum": float(np.sum(uav_exceed_boundary_penalty)),
                "uav_collision_penalty_sum": float(np.sum(uav_collision_penalty)),
                "no_solution_penalty":float(no_solution_penalty),
                "flight_energy": float(flight_energy),
                "sensing_energy": float(sensing_energy),
                "offloading_energy": float(offloading_energy),
                "computation_energy": float(computation_energy),
                "avg_uav_sen_power": float(np.mean(per_uav_sen_power_list)) if per_uav_sen_power_list else float("nan"),
                "avg_uav_off_power": float(np.mean(per_uav_off_power_list)) if per_uav_off_power_list else float("nan"),
                "avg_bs_freq": float(np.mean(per_uav_bs_freq_list)) if per_uav_bs_freq_list else float("nan"),
                "avg_uav_speed": float(np.mean(
                    np.linalg.norm(uavs_pos_cur - uavs_pos, axis=1) / self.base_args.time_slot_duration
                )),
                "avg_cu_off_power": avg_cu_off_power,
                "avg_bs_cu_freq": avg_bs_cu_freq,
            }
        }
        return float(total_reward), reward, energy_opt
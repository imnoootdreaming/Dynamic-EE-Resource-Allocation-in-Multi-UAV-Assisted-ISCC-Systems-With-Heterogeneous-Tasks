import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 内层 PC3P 由同目录桥接模块接入：它把外层实时信道/位置/动作映射为内层 P5 求解上下文。
# 函数名沿用旧接口 penalty_based_cccp，故下方调用点无需改动。
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cccp_bridge import (
    solve_inner_energy as penalty_based_cccp,
    ENERGY_CONTRACT, unavailable_energy_breakdown,
)
import numpy as np
from reward.sensing_penalty import validate_sensing_penalty_settings

SENSING_PENALTY_CONTRACT = "removed_v1"


class MyReward:
    def __init__(self, base_args, enable_diagnostics=True, *,
                 sensing_penalty_coef=0.0, sensing_penalty_reference_factor=10.0):
        self.base_args = base_args
        self.enable_diagnostics = bool(enable_diagnostics)
        (self.sensing_penalty_coef,
         self.sensing_penalty_reference_factor) = validate_sensing_penalty_settings(
             sensing_penalty_coef, sensing_penalty_reference_factor)
        if self.sensing_penalty_coef != 0.0:
            raise ValueError("sensing shortfall penalty was removed; coefficient must be 0")

    def reward_compute(self, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                       uavs_pos, uavs_pos_cur, uavs_off_duration, cus_off_power, cus_entertaining_task_size,
                       uavs_rec_beam_vectors=None, *, diagnostic_candidates=True,
                       diagnostic_cu_min_power=None, capture_failed_solution=False,
                       cu_task_cycles=None, cu_task_deadlines=None, slot_duration=None):
        tau = self.base_args.time_slot_duration if slot_duration is None else float(slot_duration)
        cycles = (np.full(self.base_args.cus_num, self.base_args.bs_cycles_per_bit)
                  if cu_task_cycles is None else np.asarray(cu_task_cycles))
        deadlines = (np.full(self.base_args.cus_num, self.base_args.cu_max_delay)
                     if cu_task_deadlines is None else np.asarray(cu_task_deadlines))
        uav_collision_penalty = np.zeros(self.base_args.uavs_num)

        for i in range(self.base_args.uavs_num):
            for j in range(i + 1, self.base_args.uavs_num):
                dist = np.linalg.norm(uavs_pos_cur[i] - uavs_pos_cur[j])
                if dist < self.base_args.uav_safe_distance:
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1

        bs_alloc_spectrum_penalty = np.sum(np.maximum(0, np.sum(uavs_cus_matched_matrix, axis=0) - 1))
        uavs_off_duration = [float(x) for x in np.asarray(uavs_off_duration).reshape(-1)]
        cus_off_power = [float(x) for x in np.asarray(cus_off_power).reshape(-1)]

        task_kwargs = {key: value for key, value in (
            ("cu_task_cycles", cu_task_cycles), ("cu_task_deadlines", cu_task_deadlines),
            ("slot_duration", slot_duration)) if value is not None}
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
            return_solution=True,
            collect_diagnostics=self.enable_diagnostics,
            diagnostic_candidates=diagnostic_candidates,
            diagnostic_cu_min_power=diagnostic_cu_min_power,
            capture_failed_solution=capture_failed_solution,
            **task_kwargs,
        )

        avg_cu_off_power = float(np.mean(cus_off_power)) if len(cus_off_power) > 0 else float("nan")
        bs_cu_freq_list = []
        if len(cur_cus_off_duration) > 0 and np.isfinite(cur_cus_off_duration[0]):
            for j in range(self.base_args.cus_num):
                denom = max(deadlines[j] - float(cur_cus_off_duration[j]), 1e-8)
                f_cu = cycles[j] * cus_entertaining_task_size[j] / denom
                bs_cu_freq_list.append(float(f_cu))
        avg_bs_cu_freq = float(np.mean(bs_cu_freq_list)) if bs_cu_freq_list else float("nan")

        # Single-source six-component accounting from the inner objective.
        # A missing/failed service solve does not erase known flight energy.
        energy_breakdown = (solution_payload or {}).get("energy_breakdown")
        if not isinstance(energy_breakdown, dict):
            uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                known_flight = float(np.sum(
                    self.base_args.uav_c1 * (uav_dist_diff ** 3) / (tau ** 2)
                    + self.base_args.uav_c2 * (tau ** 2) / uav_dist_diff
                ))
            energy_breakdown = {
                **unavailable_energy_breakdown(known_flight),
                "energy_contract": ENERGY_CONTRACT,
                "energy_solution_status": "missing_breakdown",
                "energy_finite_candidate": bool(np.isfinite(energy_opt)),
                "energy_physical_solution_verified": None,
                "energy_strict_feasible": None,
            }
        else:
            energy_breakdown = dict(energy_breakdown)

        # Legacy names remain available, with the BS sensing computation bug
        # fixed. Canonical *_j fields explicitly identify device/task and units.
        def legacy_component(key):
            value = energy_breakdown.get(key)
            return float(value) if value is not None else float("nan")

        flight_energy = legacy_component("uav_flight_energy_j")
        sensing_energy = legacy_component("uav_sensing_energy_j")
        offloading_energy = legacy_component("uav_offloading_energy_j")
        computation_energy = legacy_component("bs_sensing_computation_energy_j")

        no_solution_penalty = 0
        if energy_opt == float("inf"):
            no_solution_penalty = 1

        total_reward_4_energy = np.exp(-energy_opt / 1000)
        total_reward = (
            total_reward_4_energy
            - bs_alloc_spectrum_penalty
            - np.sum(uav_collision_penalty)
            - no_solution_penalty
        )

        base_reward = float(total_reward)
        base_no_solution_penalty = no_solution_penalty

        reward = {
            "bs": float(total_reward),
            "components": {
                "total_reward_4_energy": float(total_reward_4_energy),
                "bs_alloc_spectrum_penalty": float(bs_alloc_spectrum_penalty),
                "uav_collision_penalty_sum": float(np.sum(uav_collision_penalty)),
                "no_solution_penalty":float(no_solution_penalty),
                "base_reward": base_reward,
                "base_no_solution_penalty": float(base_no_solution_penalty),
                "sensing_penalty_contract": SENSING_PENALTY_CONTRACT,
                "sensing_penalty_score": 0.0,
                "sensing_penalty_applied": 0.0,
                "sensing_penalty_per_uav": None,
                "sensing_margin": None,
                "sensing_penalty_status": "removed",
                "sensing_penalty_reason": None,
                "flight_energy": float(flight_energy),
                "sensing_energy": float(sensing_energy),
                "offloading_energy": float(offloading_energy),
                "computation_energy": float(computation_energy),
                **energy_breakdown,
                "avg_uav_sen_power": float(np.mean(per_uav_sen_power_list)) if per_uav_sen_power_list else float("nan"),
                "avg_uav_off_power": float(np.mean(per_uav_off_power_list)) if per_uav_off_power_list else float("nan"),
                "avg_bs_freq": float(np.mean(per_uav_bs_freq_list)) if per_uav_bs_freq_list else float("nan"),
                "avg_uav_speed": float(np.mean(
                    np.linalg.norm(uavs_pos_cur - uavs_pos, axis=1) / tau
                )),
                "avg_cu_off_power": avg_cu_off_power,
                "avg_bs_cu_freq": avg_bs_cu_freq,
                "solver_status": str((solution_payload or {}).get("solver_status", "unknown")),
                "solver_status_kind": str((solution_payload or {}).get("solver_status_kind", "other")),
                **{key: (solution_payload or {})[key] for key in (
                    "raw_solver_status", "raw_solver_status_kind", "raw_problem_status",
                    "status_source", "optimizer_response_code", "infeasibility_certificate",
                    "presolve_contract", "solver_skipped", "pc3p_called", "presolve_seconds",
                    "presolve_certificate",
                ) if key in (solution_payload or {})},
                "solver_converged": bool((solution_payload or {}).get("solver_converged", False)),
                "solver_iterations": int((solution_payload or {}).get("solver_iterations", 0) or 0),
            }
        }
        if self.enable_diagnostics:
            reward["diagnostics"] = (solution_payload or {}).get("env_diagnostic")
        return float(total_reward), reward, energy_opt

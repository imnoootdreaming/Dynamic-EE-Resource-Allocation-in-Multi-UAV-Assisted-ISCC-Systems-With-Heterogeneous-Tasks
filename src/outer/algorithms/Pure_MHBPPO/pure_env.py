"""Current scenario/observation with legacy Pure actions and direct P0 physics."""
import numpy as np
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from environment.my_env import MyEnv
from environment.mapping import build_uavs_cus_matched_matrix, build_uav_targets_matched_matrix
from algorithms.Pure_MHBPPO.physical import evaluate_physical


class PureMyEnv(MyEnv):
    def __init__(self, base_args, madrl_args, seed_offset=0):
        # Base initialization imports/constructs MyReward, but Pure.step never
        # invokes it. Existing MHBPPO environment behaviour remains unchanged.
        super().__init__(base_args, madrl_args, seed_offset=seed_offset)
        I, J, N = self.base_args.uavs_num, self.base_args.cus_num, self.base_args.antenna_nums
        self.bs_continuous_action_splits = {
            "uav_angles": I, "uav_distances": I, "uav_off_durations": I,
            "uav_sen_beam_dir_real": I*N, "uav_sen_beam_dir_imag": I*N,
            "uav_sen_beam_power": I, "uav_off_beam_dir_real": I*N,
            "uav_off_beam_dir_imag": I*N, "uav_off_beam_power": I,
            "cu_off_powers": J, "cu_off_durations": J,
        }
        low, high = self._bounds(float(np.min(self.cu_tasks.slot_seconds)),
                                 np.min(self.cu_tasks.d_seconds, axis=0))
        _, high_envelope = self._bounds(float(np.max(self.cu_tasks.slot_seconds)),
                                         np.max(self.cu_tasks.d_seconds, axis=0))
        # max speed depends on slot; other high bounds use episode envelope.
        high = np.maximum(high, high_envelope)
        self.action_space = {"bs": {"continuous": spaces.Box(low=low, high=high, dtype=np.float32),
                                    "discrete": spaces.MultiDiscrete(np.full(I, J, dtype=np.int64))}}

    def _bounds(self, slot, deadlines):
        a = self.base_args
        I, J, N = a.uavs_num, a.cus_num, a.antenna_nums
        eps = float(self.epsilon)
        pmax = 10 ** ((float(a.cu_max_power_dbm)-30)/10)
        low = np.concatenate([np.zeros(I), np.full(I, a.uav_min_speed*slot), np.full(I, eps),
                              np.full(I*N, -1), np.full(I*N, -1), np.zeros(I),
                              np.full(I*N, -1), np.full(I*N, -1), np.zeros(I),
                              np.full(J, eps), np.full(J, eps)]).astype(np.float32)
        high = np.concatenate([np.full(I, 2*np.pi), np.full(I, a.uav_max_speed*slot),
                               np.full(I, a.uav_max_delay-a.uav_sen_duration),
                               np.ones(I*N), np.ones(I*N), np.ones(I),
                               np.ones(I*N), np.ones(I*N), np.ones(I),
                               np.full(J, pmax), np.asarray(deadlines)]).astype(np.float32)
        if np.any(high <= low):
            raise ValueError("Pure action interval is empty under current physical parameters")
        return low, high

    def get_current_action_bounds(self):
        return self._bounds(float(self.current_slot_duration), self.cus_task_deadlines)

    @staticmethod
    def _beams(real, imag, ratios, maximum):
        direction = real + 1j*imag
        norms = np.linalg.norm(direction, axis=1)
        result = np.empty_like(direction, dtype=np.complex128)
        for i in range(len(norms)):
            d = direction[i] if norms[i] >= 1e-12 else np.eye(1, direction.shape[1], 0, dtype=complex)[0]
            norm = norms[i] if norms[i] >= 1e-12 else 1.0
            result[i] = d / norm * np.sqrt(np.clip(ratios[i], 0, 1)*maximum)
        return result

    def step(self, actions, i_episode=None):
        if self.t >= self.madrl_args.total_time_slots:
            raise RuntimeError("episode has terminated; reset before stepping again")
        action = actions["bs"] if isinstance(actions, dict) and "bs" in actions else actions
        raw = np.asarray(action["continuous"], dtype=np.float32)
        discrete = np.asarray(action["discrete"], dtype=np.int64)
        I, J, N = self.base_args.uavs_num, self.base_args.cus_num, self.base_args.antenna_nums
        if raw.shape != self.action_space["bs"]["continuous"].shape or discrete.shape != (I,):
            raise ValueError("Pure action shape mismatch")
        if not np.all(np.isfinite(raw)) or np.any(discrete < 0) or np.any(discrete >= J):
            raise ValueError("Pure action is nonfinite or CU index outside range")
        low, high = self.get_current_action_bounds()
        normalized = np.clip(raw, 0, 1)
        mapped = normalized*(high-low)+low
        offset = 0
        def take(n):
            nonlocal offset
            value = mapped[offset:offset+n]
            offset += n
            return value
        angle, distance, du = take(I), take(I), take(I)
        w_real, w_imag, w_ratio = take(I*N).reshape(I, N), take(I*N).reshape(I, N), take(I)
        b_real, b_imag, b_ratio = take(I*N).reshape(I, N), take(I*N).reshape(I, N), take(I)
        p, dc = take(J), take(J)
        w = self._beams(w_real, w_imag, w_ratio, self.base_args.uav_max_power)
        b = self._beams(b_real, b_imag, b_ratio, self.base_args.uav_max_power)
        eta = build_uavs_cus_matched_matrix(discrete, I, J)
        before = self.cur_uavs_pos.copy()
        after = before + np.stack([distance*np.cos(angle), distance*np.sin(angle), np.zeros(I)], axis=1)
        target_indices = self._get_current_uav_target_indices()
        A = self.uavs_2_targets_channels[np.arange(I), target_indices]
        physical = evaluate_physical(params=self.base_args, eta=eta, p_cu=p, duration_uav=du, duration_cu=dc,
            sensing_beams=w, offloading_beams=b, positions_before=before, positions_after=after,
            h_cu_uav=self.uavs_2_cus_channels, h_uav_bs=self.uavs_2_bs_channels,
            h_cu_bs=self.cus_2_bs_channels, sensing_channels=A, task_bits=self.cus_entertaining_task_size,
            task_cycles=self.cus_task_cycles, task_deadlines=self.cus_task_deadlines,
            slot_seconds=float(self.current_slot_duration))
        success = int(physical["feasible"])
        reward = float(physical["reward"])
        energy = physical["candidate_energy"]
        self.last_action_trace = {"time_slot": int(self.t), "raw_continuous": raw.tolist(),
            "discrete": discrete.tolist(), "mapped_continuous": mapped.tolist(),
            "continuous_low": low.tolist(), "continuous_high": high.tolist(),
            "position_before": before.tolist(), "position_after": after.tolist(),
            "sensing_beams": {"real": w.real.tolist(), "imag": w.imag.tolist()},
            "offloading_beams": {"real": b.real.tolist(), "imag": b.imag.tolist()},
            "cu_tasks": self._current_task_record(), "physical": physical}
        self.last_env_diagnostic = physical
        self._episode_all_success = self._episode_all_success and bool(success)
        reward_dict = {"bs": reward, "components": {"backend": "direct_no_optimizer",
            "solver_status_kind": "not_called", "candidate_feasible": success,
            "candidate_energy": energy, "constraint_checks": physical["checks"],
            "energy_breakdown": physical.get("energy"),
            "bs_alloc_spectrum_penalty": physical.get("spectrum_penalty", 0.0),
            "uav_collision_penalty_sum": physical.get("collision_penalty", 0.0),
            "candidate_constraint_penalty": physical.get("service_candidate_penalty", 1)}}
        self.t += 1
        self._select_current_tasks()
        self.uav_previous_displacement = after - before
        self.cur_uavs_pos = after
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self._refresh_channels()
        self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t], uavs_num=I, targets_num=self.base_args.targets_num)
        return ({"bs": self._build_bs_observation()}, reward, reward_dict,
                int(self.t >= self.madrl_args.total_time_slots),
                float(energy) if energy is not None else float("inf"), success)

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
import numpy as np

from reward.my_reward import MyReward


def dbm_2_watt(dbm):
    return 10 ** ((dbm - 30) / 10)


def db_2_watt(db):
    return 10 ** (db / 10)


class MyEnv(gym.Env):
    def __init__(self, base_args, madrl_args):
        super(MyEnv, self).__init__()
        self.base_args = base_args
        self.madrl_args = madrl_args
        self.epsilon = 1e-4
        self.t = 0
        self.target_hold_slots = 4  # 每 4 个时隙分配一次目标
        # 固定 CU 的任务量
        self.cus_entertaining_task_size = np.ones(self.base_args.cus_num) * 170e3
        # 生成初始 UAV / CU / 目标位置，并预计算 CU 轨迹和 UAV-目标分配调度
        self.init_uavs_pos, self.init_cus_pos, self.init_targets_pos = self.generate_pos(
            self.base_args.uavs_num,
            self.base_args.cus_num,
            self.base_args.targets_num,
            self.base_args.center,
            self.base_args.radius,
            self.base_args.uav_height
        )
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.precomputed_cus_traj = self.generate_cu_trajectory()
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        # 20260404 - NLoS 分量: 预生成每个时隙的高斯散射分量，保证不同 episode 的相同时隙复用同一 realization
        self.precomputed_nlos_components = self._precompute_nlos_components()

        self.precomputed_uav_target_schedule, self.precomputed_uav_target_schedule_distances = self.generate_uav_target_schedule()
        self._print_precomputed_target_schedule(
            schedule=self.precomputed_uav_target_schedule,
            schedule_distances=self.precomputed_uav_target_schedule_distances,
            hold_slots=self.target_hold_slots,
            total_time_slots=self.madrl_args.total_time_slots
        )
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )

        self._refresh_channels()

        self.bs_continuous_action_splits = {
            "uav_angles": self.base_args.uavs_num,
            "uav_distances": self.base_args.uavs_num,
            "uav_off_durations": self.base_args.uavs_num,
            "cu_off_powers": self.base_args.cus_num,
        }
        # NOTE - UAV 个离散头: 每个 UAV 选择一个 CU 索引进行匹配
        self.bs_discrete_action_dims = np.full(self.base_args.uavs_num, self.base_args.cus_num, dtype=np.int64)

        bs_continuous_low = np.concatenate([
            np.zeros(self.base_args.uavs_num, dtype=np.float32),  # UAV 飞行角度
            np.full(
                self.base_args.uavs_num,
                self.base_args.uav_min_speed * self.base_args.time_slot_duration,
                dtype=np.float32
            ), # UAV 飞行距离
            np.full(self.base_args.uavs_num, self.epsilon, dtype=np.float32),  # UAV 卸载时长
            np.full(self.base_args.cus_num, self.epsilon, dtype=np.float32),  # CU 卸载功率
        ])
        bs_continuous_high = np.concatenate([
            np.full(self.base_args.uavs_num, 2 * np.pi, dtype=np.float32),   # UAV 飞行角度
            np.full(
                self.base_args.uavs_num,
                self.base_args.uav_max_speed * self.base_args.time_slot_duration,
                dtype=np.float32
            ),  # UAV 飞行距离
            np.full(
                self.base_args.uavs_num,
                self.base_args.uav_max_delay - self.base_args.uav_sen_duration,
                dtype=np.float32
            ),  # UAV 卸载时长
            np.full(self.base_args.cus_num, dbm_2_watt(self.base_args.cu_max_power_dbm), dtype=np.float32),  # CU 卸载功率
        ])

        self.action_space = {
            "bs": {
                "continuous": spaces.Box(low=bs_continuous_low, high=bs_continuous_high, dtype=np.float32),
                "discrete": spaces.MultiDiscrete(self.bs_discrete_action_dims),
            }
        }

        # BS 观测空间
        # 1. 时隙 t UAV-BS 信道
        # 2. 时隙 t UAV-CU 信道
        # 3. 时隙 t CU-BS 信道
        # 4. 时隙 t UAV感知目标位置
        # 5. 时隙 t UAV 位置
        # 6. 窗口倒计时
        # 7. 当前窗口每个 UAV 对应 target 的坐标
        # 8. 下一窗口每个 UAV 对应 target 的坐标
        obs_dim_bs = (
            self.base_args.uavs_num * self.base_args.antenna_nums * 2
            + self.base_args.uavs_num * self.base_args.cus_num * self.base_args.antenna_nums * 2
            + self.base_args.cus_num * 2
            + self.base_args.targets_num * 3
            + self.base_args.uavs_num * 3
            #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
            + 1
            + self.base_args.uavs_num * 3
            + self.base_args.uavs_num * 3
        )
        self.observation_space = {
            "bs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_bs,), dtype=np.float32)
        }

        self.reward_calculator = MyReward(self.base_args)

    def _precompute_nlos_components(self):
        # 20260404 - NLoS 分量: 以固定 seed 按时隙预生成通信链路的 NLoS 高斯样本
        total_slots = self.madrl_args.total_time_slots + 1
        rng = np.random.default_rng(self.base_args.seed)
        return {
            "uavs_2_cus": (
                rng.standard_normal((total_slots, self.base_args.uavs_num, self.base_args.cus_num, self.base_args.antenna_nums))
                + 1j * rng.standard_normal((total_slots, self.base_args.uavs_num, self.base_args.cus_num, self.base_args.antenna_nums))
            ) / np.sqrt(2),
            "uavs_2_bs": (
                rng.standard_normal((total_slots, self.base_args.uavs_num, 1, self.base_args.antenna_nums))
                + 1j * rng.standard_normal((total_slots, self.base_args.uavs_num, 1, self.base_args.antenna_nums))
            ) / np.sqrt(2),
            "cus_2_bs": (
                rng.standard_normal((total_slots, self.base_args.cus_num, 1))
                + 1j * rng.standard_normal((total_slots, self.base_args.cus_num, 1))
            ) / np.sqrt(2),
        }

    def _refresh_channels(self):
        # 20260404 - NLoS 分量: 当前时隙固定使用预生成的 NLoS 样本
        slot_nlos = {
            "uavs_2_cus": self.precomputed_nlos_components["uavs_2_cus"][self.t],
            "uavs_2_bs": self.precomputed_nlos_components["uavs_2_bs"][self.t],
            "cus_2_bs": self.precomputed_nlos_components["cus_2_bs"][self.t],
        }
        self.uavs_2_cus_channels, self.uavs_2_bs_channels, self.cus_2_bs_channels = self.compute_com_channel_gain(
            uavs_pos=self.cur_uavs_pos,
            cus_pos=self.cur_cus_pos,
            ref_path_loss=db_2_watt(self.base_args.ref_path_loss_db),
            frac_d_lambda=self.base_args.frac_d_lambda,
            alpha_uav_link=self.base_args.alpha_uav_link,
            alpha_cu_link=self.base_args.alpha_cu_link,
            rician_factor=db_2_watt(self.base_args.rician_factor_db),
            antenna_nums=self.base_args.antenna_nums,
            nlos_components=slot_nlos
        )
        self.uavs_2_targets_channels = self.compute_sen_channel_gain(
            radar_rcs=self.base_args.radar_rcs,
            frac_d_lambda=self.base_args.frac_d_lambda,
            uavs_pos=self.cur_uavs_pos,
            targets_pos=self.init_targets_pos,
            antenna_nums=self.base_args.antenna_nums,
            ref_path_loss=db_2_watt(self.base_args.ref_path_loss_db)
        )

    def _flatten_complex(self, channel):
        return np.concatenate([channel.real.flatten(), channel.imag.flatten()]).astype(np.float32)

    def _get_all_target_coords(self):
        return self.init_targets_pos.astype(np.float32).flatten()

    def _get_slots_until_switch(self):
        #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
        effective_t = min(self.t, max(self.madrl_args.total_time_slots - 1, 0))
        slots_until_switch = self.target_hold_slots - 1 - (effective_t % self.target_hold_slots)
        return np.array([slots_until_switch], dtype=np.float32)

    def _get_target_coords_by_indices(self, target_indices):
        clipped_target_indices = np.clip(
            np.asarray(target_indices, dtype=np.int64),
            0,
            self.base_args.targets_num - 1
        )
        return self.init_targets_pos[clipped_target_indices].astype(np.float32).flatten()

    def _get_current_uav_target_coords(self):
        effective_t = min(self.t, self.madrl_args.total_time_slots)
        return self._get_target_coords_by_indices(
            self.precomputed_uav_target_schedule[effective_t]
        )

    def _get_next_uav_target_coords(self):
        #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
        if self.madrl_args.total_time_slots <= 0:
            return self._get_current_uav_target_coords()
        # 防止最后一个窗口的越界保护
        effective_t = min(self.t, self.madrl_args.total_time_slots - 1)
        # 计算总共有多少个目标分配窗口
        num_windows = self._get_num_target_windows(
            total_time_slots=self.madrl_args.total_time_slots,
            hold_slots=self.target_hold_slots
        )
        # 计算当前时隙属于第几个窗口
        current_window_idx = min(effective_t // self.target_hold_slots, num_windows - 1)
        # 计算下一个窗口
        next_window_idx = min(current_window_idx + 1, num_windows - 1)
        # 计算下一个窗口从哪个时隙开始
        next_window_start_slot = min(next_window_idx * self.target_hold_slots, self.madrl_args.total_time_slots)
        return self._get_target_coords_by_indices(
            self.precomputed_uav_target_schedule[next_window_start_slot]
        )

    def _build_bs_observation(self):
        return np.concatenate([
            self._flatten_complex(self.uavs_2_bs_channels),
            self._flatten_complex(self.uavs_2_cus_channels),
            self._flatten_complex(self.cus_2_bs_channels),
            self._get_all_target_coords(),
            self.cur_uavs_pos.astype(np.float32).flatten(),
            #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
            self._get_slots_until_switch(),
            self._get_current_uav_target_coords(),
            self._get_next_uav_target_coords(),
        ]).astype(np.float32)

    def _build_uavs_cus_matched_matrix(self, discrete_actions):
        discrete_actions = np.clip(np.asarray(discrete_actions, dtype=np.int64), 0, self.base_args.cus_num - 1)
        uavs_cus_matched_matrix = np.zeros((self.base_args.uavs_num, self.base_args.cus_num), dtype=np.float32)
        for uav_idx, cu_idx in enumerate(discrete_actions):
            uavs_cus_matched_matrix[uav_idx, int(cu_idx)] = 1.0
        return uavs_cus_matched_matrix

    def step(self, actions, i_episode=None):
        bs_actions = actions["bs"] if isinstance(actions, dict) and "bs" in actions else actions

        continuous_actions = np.asarray(bs_actions["continuous"], dtype=np.float32)
        discrete_actions = np.asarray(bs_actions["discrete"], dtype=np.int64)

        continuous_low = self.action_space["bs"]["continuous"].low
        continuous_high = self.action_space["bs"]["continuous"].high
        continuous_actions = np.clip(continuous_actions, 0.0, 1.0)
        continuous_actions = continuous_actions * (continuous_high - continuous_low) + continuous_low

        offset = 0
        diff_theta = continuous_actions[offset:offset + self.base_args.uavs_num]
        offset += self.base_args.uavs_num
        diff_distance = continuous_actions[offset:offset + self.base_args.uavs_num]
        offset += self.base_args.uavs_num
        off_duration = continuous_actions[offset:offset + self.base_args.uavs_num]
        offset += self.base_args.uavs_num
        cus_off_power = continuous_actions[offset:offset + self.base_args.cus_num]

        # NOTE - 将离散 CU 索引动作转换为 UAV-CU 匹配矩阵输入 CCCP，以适配 reward 计算接口
        uavs_cus_matched_matrix = self._build_uavs_cus_matched_matrix(discrete_actions)

        next_uavs_pos = self.cur_uavs_pos + np.stack([
            diff_distance * np.cos(diff_theta),
            diff_distance * np.sin(diff_theta),
            np.zeros_like(diff_distance),
        ], axis=1)

        color_reset = "\033[0m"
        color_title = "\033[38;5;67m"
        color_metric = "\033[38;5;109m"
        color_energy = "\033[38;5;108m"
        color_penalty = "\033[38;5;137m"
        print(
            f"=================================================="
        )
        # ---- Action Details ----
        print(f"{color_title} ------------ [Action Details] ------------ {color_reset}")
        for i in range(self.base_args.uavs_num):
            print(
                f"  {color_metric}UAV-{i}{color_reset}: "
                f"angle={diff_theta[i]:.4f} rad, "
                f"dist={diff_distance[i]:.4f} m, "
                f"off_duration={off_duration[i]:.4f} s, "
                f"{color_energy}→ CU-{int(discrete_actions[i])}{color_reset}"
            )
        for i in range(self.base_args.cus_num):
            print(
                f"  {color_penalty}CU-{i}{color_reset}: "
                f"off_power={cus_off_power[i]:.4f} W"
            )
        
        total_reward, reward, energy_opt = self.reward_calculator.reward_compute(
            uavs_2_cus_channels=self.uavs_2_cus_channels,
            uavs_2_bs_channels=self.uavs_2_bs_channels,
            cus_2_bs_channels=self.cus_2_bs_channels,
            uavs_2_targets_channels=self.uavs_2_targets_channels,
            uavs_targets_matched_matrix=self.uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched_matrix,
            uavs_pos=self.cur_uavs_pos,
            uavs_pos_cur=next_uavs_pos,
            uavs_off_duration=off_duration,
            cus_off_power=cus_off_power,
            cus_entertaining_task_size=self.cus_entertaining_task_size
        )
        print(
            f"{color_title}Episode {i_episode}, Time Slot {self.t}:{color_reset} "
            f"{color_metric}Total Reward = {total_reward:.4f}{color_reset}, "
            f"{color_energy}Energy Opt = {energy_opt:.4f}{color_reset}"
        )
        # ---- Reward Details ----
        reward_components = reward.get("components", {})
        print(f"{color_title} ------------ [Reward Details] ------------ {color_reset}")
        print(f"  {color_metric}BS Reward{color_reset}: {reward.get('bs', 0.0):.4f}")
        print(f"  {color_energy}total_reward_4_energy{color_reset}: {reward_components.get('total_reward_4_energy', 0.0):.4f}")
        print(f"  {color_penalty}bs_alloc_spectrum_penalty{color_reset}: {reward_components.get('bs_alloc_spectrum_penalty', 0.0):.4f}")
        print(f"  {color_penalty}uav_exceed_boundary_penalty_sum{color_reset}: {reward_components.get('uav_exceed_boundary_penalty_sum', 0.0):.4f}")
        print(f"  {color_penalty}uav_collision_penalty_sum{color_reset}: {reward_components.get('uav_collision_penalty_sum', 0.0):.4f}")
        print(f"  {color_penalty}no_solution_penalty{color_reset}: {reward_components.get('no_solution_penalty', 0.0):.4f}")
        print(
            f"=================================================="
        )
        # 判断任务是否完成
        success = int(
            reward_components.get("no_solution_penalty", 0) == 0 and
            reward_components.get("uav_collision_penalty_sum", 0) == 0 and
            reward_components.get("uav_exceed_boundary_penalty_sum", 0) == 0 and
            reward_components.get("bs_alloc_spectrum_penalty", 0) == 0
        )
        self.t += 1
        self.cur_uavs_pos = next_uavs_pos
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self._refresh_channels()
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )

        next_state_dict = {"bs": self._build_bs_observation()}
        done = int(self.t >= self.madrl_args.total_time_slots)
        return next_state_dict, float(total_reward), reward, done, energy_opt, success

    def reset(self):
        self.t = 0
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )
        self._refresh_channels()
        return {"bs": self._build_bs_observation()}

    def getPosUAV(self):
        return self.cur_uavs_pos.copy()

    def getPosCU(self):
        return self.cur_cus_pos.copy()

    def getPosTarget(self):
        return self.init_targets_pos.copy()

    def _get_num_target_windows(self, total_time_slots, hold_slots):
        return int(np.ceil(total_time_slots / hold_slots))

    def _validate_target_schedule_requirements(self, total_time_slots, hold_slots):
        required_target_num = self._get_num_target_windows(total_time_slots, hold_slots) * self.base_args.uavs_num
        if self.base_args.targets_num != required_target_num:
            raise ValueError(
                "The current sensing schedule requires "
                f"targets_num == ceil(total_time_slots / hold_slots) * uavs_num. "
                f"Got targets_num={self.base_args.targets_num}, required={required_target_num}, "
                f"hold_slots={hold_slots}, total_time_slots={total_time_slots}."
            )
        return required_target_num

    def _assign_targets_for_window(self, reference_positions, candidate_target_indices, target_positions=None):
        if target_positions is None:
            target_positions = self.init_targets_pos
        pending_uavs = list(range(self.base_args.uavs_num))
        pending_targets = list(candidate_target_indices)
        assigned_targets = np.full(self.base_args.uavs_num, -1, dtype=np.int64)
        assigned_distances = np.zeros(self.base_args.uavs_num, dtype=np.float32)

        while pending_uavs:
            best_distance = None
            best_uav_idx = None
            best_target_idx = None
            best_target_list_idx = None

            for uav_idx in pending_uavs:
                candidate_targets = np.asarray(pending_targets, dtype=np.int64)
                candidate_positions = target_positions[candidate_targets]
                candidate_distances = np.linalg.norm(candidate_positions - reference_positions[uav_idx], axis=1)
                nearest_local_idx = int(np.argmin(candidate_distances))
                nearest_distance = float(candidate_distances[nearest_local_idx])
                nearest_target_idx = int(candidate_targets[nearest_local_idx])

                if best_distance is None or nearest_distance < best_distance:
                    best_distance = nearest_distance
                    best_uav_idx = uav_idx
                    best_target_idx = nearest_target_idx
                    best_target_list_idx = nearest_local_idx

            assigned_targets[best_uav_idx] = best_target_idx
            assigned_distances[best_uav_idx] = float(best_distance)
            pending_uavs.remove(best_uav_idx)
            pending_targets.pop(best_target_list_idx)

        return assigned_targets, assigned_distances

    def generate_uav_target_schedule(self, hold_slots=None, total_time_slots=None, initial_reference_positions=None):
        if hold_slots is None:
            hold_slots = self.target_hold_slots
        if total_time_slots is None:
            total_time_slots = self.madrl_args.total_time_slots

        schedule = np.zeros((total_time_slots + 1, self.base_args.uavs_num), dtype=np.int64)
        schedule_distances = np.zeros((total_time_slots + 1, self.base_args.uavs_num), dtype=np.float32)

        if total_time_slots <= 0:
            return schedule, schedule_distances

        self._validate_target_schedule_requirements(total_time_slots=total_time_slots, hold_slots=hold_slots)
        remaining_targets = list(range(self.base_args.targets_num))
        if initial_reference_positions is None:
            reference_positions = self.init_uavs_pos.copy()
        else:
            reference_positions = np.asarray(initial_reference_positions, dtype=float).copy()

        for window_idx in range(self._get_num_target_windows(total_time_slots, hold_slots)):
            assigned_targets, assigned_distances = self._assign_targets_for_window(
                reference_positions=reference_positions,
                candidate_target_indices=remaining_targets
            )
            slot_start = window_idx * hold_slots
            slot_end = min(slot_start + hold_slots, total_time_slots)
            schedule[slot_start:slot_end] = assigned_targets
            schedule_distances[slot_start:slot_end] = assigned_distances

            assigned_target_set = set(assigned_targets.tolist())
            remaining_targets = [target_idx for target_idx in remaining_targets if target_idx not in assigned_target_set]
            reference_positions = self.init_targets_pos[assigned_targets].copy()

        schedule[total_time_slots] = schedule[total_time_slots - 1]
        schedule_distances[total_time_slots] = schedule_distances[total_time_slots - 1]
        return schedule, schedule_distances

    def _print_precomputed_target_schedule(self, schedule, schedule_distances, hold_slots, total_time_slots):
        print("==================================================")
        print("------------ [Precomputed Target Schedule] ------------")
        for t in range(total_time_slots):
            window_idx = t // hold_slots
            window_start = window_idx * hold_slots + 1
            window_end = min((window_idx + 1) * hold_slots, total_time_slots)
            assignment_parts = []
            for uav_idx in range(self.base_args.uavs_num):
                target_idx = int(schedule[t, uav_idx])
                transition_distance = float(schedule_distances[t, uav_idx])
                assignment_parts.append(
                    f"UAV-{uav_idx}->Target-{target_idx} (planned_transition_distance={transition_distance:.4f} m)"
                )
            print(
                f"Time Slot {t + 1:02d}/{total_time_slots} | "
                f"Window {window_idx + 1} ({window_start}-{window_end}) | "
                + " | ".join(assignment_parts)
            )
        print("==================================================")

    def build_uav_targets_matched_matrix(self, target_indices):
        matched_matrix = np.zeros((self.base_args.uavs_num, self.base_args.targets_num), dtype=np.float32)
        clipped_target_indices = np.clip(
            np.asarray(target_indices, dtype=np.int64),
            0,
            self.base_args.targets_num - 1
        )
        for i, target_idx in enumerate(clipped_target_indices):
            matched_matrix[i, target_idx] = 1.0
        return matched_matrix

    def generate_pos(self, uavs_num, cus_num, targets_num, center, radius, uav_height):
        r_cu = radius * np.sqrt(np.random.rand(cus_num))
        theta_cu = np.random.rand(cus_num) * 2 * np.pi
        cus_pos = np.zeros((cus_num, 3))
        cus_pos[:, 0] = center[0] + r_cu * np.cos(theta_cu)
        cus_pos[:, 1] = center[1] + r_cu * np.sin(theta_cu)

        r_target = radius * np.sqrt(np.random.rand(targets_num))
        theta_target = np.random.rand(targets_num) * 2 * np.pi
        targets_pos = np.zeros((targets_num, 3))
        targets_pos[:, 0] = center[0] + r_target * np.cos(theta_target)
        targets_pos[:, 1] = center[1] + r_target * np.sin(theta_target)

        initial_target_indices = np.random.choice(
            targets_num,
            size=uavs_num,
            replace=targets_num < uavs_num
        )
        uavs_pos = np.zeros((uavs_num, 3))
        uavs_pos[:, :2] = targets_pos[initial_target_indices, :2]
        uavs_pos[:, 2] = uav_height

        return uavs_pos, cus_pos, targets_pos

    def compute_com_channel_gain(self, uavs_pos, cus_pos, ref_path_loss, frac_d_lambda,
                                 alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums,
                                 nlos_components=None):
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

    def compute_sen_channel_gain(self, radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
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

    def generate_cu_trajectory(self):
        traj = np.zeros((self.madrl_args.total_time_slots + 1, self.base_args.cus_num, 3), dtype=float)
        velocities = np.zeros((self.madrl_args.total_time_slots + 1, self.base_args.cus_num, 2), dtype=float)

        if self.init_cus_pos.shape[0] > 0:
            traj[0] = self.init_cus_pos.copy()

        v_init = np.array(self.base_args.markov_velocity)[:2]
        velocities[0] = np.tile(v_init, (self.base_args.cus_num, 1))
        rng = np.random.default_rng(self.base_args.seed)

        for t in range(self.madrl_args.total_time_slots):
            random_component = rng.normal(size=(self.base_args.cus_num, 2))
            v_bar = np.array(self.base_args.markov_asymptotic_mean_of_velocity)[:2]
            velocities[t + 1] = (
                self.base_args.markov_memory_level * velocities[t]
                + (1 - self.base_args.markov_memory_level) * v_bar
                + np.sqrt(1 - self.base_args.markov_memory_level ** 2)
                * self.base_args.markov_standard_deviation_of_velocity
                * random_component
            )
            traj[t + 1, :, :2] = traj[t, :, :2] + velocities[t] * self.base_args.time_slot_duration
            traj[t + 1, :, 2] = traj[t, :, 2]

        return traj

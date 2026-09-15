try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
import numpy as np

from environment.channel_models import (
    compute_com_channel_gain,
    compute_sen_channel_gain,
    db_2_watt,
    dbm_2_watt,
)
from environment.mapping import (
    build_unit_norm_rec_beam,
    build_uav_targets_matched_matrix,
    build_uavs_cus_matched_matrix,
    flatten_complex,
)
from environment.topology import (
    generate_cu_trajectory,
    generate_pos,
    generate_uav_target_schedule,
    get_num_target_windows,
    print_precomputed_target_schedule,
)
from reward.my_reward import MyReward


class MyEnv(gym.Env):
    """外层 ISCC 场景环境（MHBPPO 单智能体 BS 视角）。

    职责边界：本类只保留环境状态机（时隙 `t`、位置状态）、动作/观测空间、`reset` /
    `step` 与观测拼接。与状态机无关的通用计算已抽离：
    - 信道与单位换算 → `environment.channel_models`
    - 位置/轨迹/目标调度 → `environment.topology`
    - 匹配矩阵/接收波束/扁平化 → `environment.mapping`
    """

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
        self.init_uavs_pos, self.init_cus_pos, self.init_targets_pos = generate_pos(
            self.base_args.uavs_num,
            self.base_args.cus_num,
            self.base_args.targets_num,
            self.base_args.center,
            self.base_args.radius,
            self.base_args.uav_height
        )
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.precomputed_cus_traj = generate_cu_trajectory(
            init_cus_pos=self.init_cus_pos,
            cus_num=self.base_args.cus_num,
            total_time_slots=self.madrl_args.total_time_slots,
            markov_velocity=self.base_args.markov_velocity,
            markov_memory_level=self.base_args.markov_memory_level,
            markov_asymptotic_mean_of_velocity=self.base_args.markov_asymptotic_mean_of_velocity,
            markov_standard_deviation_of_velocity=self.base_args.markov_standard_deviation_of_velocity,
            time_slot_duration=self.base_args.time_slot_duration,
            seed=self.base_args.seed,
        )
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        # 20260404 - NLoS 分量: 预生成每个时隙的高斯散射分量，保证不同 episode 的相同时隙复用同一 realization
        self.precomputed_nlos_components = self._precompute_nlos_components()

        self.precomputed_uav_target_schedule, self.precomputed_uav_target_schedule_distances = generate_uav_target_schedule(
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
            init_uavs_pos=self.init_uavs_pos,
            init_targets_pos=self.init_targets_pos,
            total_time_slots=self.madrl_args.total_time_slots,
            hold_slots=self.target_hold_slots,
        )
        print_precomputed_target_schedule(
            uavs_num=self.base_args.uavs_num,
            schedule=self.precomputed_uav_target_schedule,
            schedule_distances=self.precomputed_uav_target_schedule_distances,
            hold_slots=self.target_hold_slots,
            total_time_slots=self.madrl_args.total_time_slots
        )
        self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t],
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
        )

        self._refresh_channels()

        self.bs_continuous_action_splits = {
            "uav_angles": self.base_args.uavs_num,
            "uav_distances": self.base_args.uavs_num,
            "uav_off_durations": self.base_args.uavs_num,
            "cu_off_powers": self.base_args.cus_num,
            # 20260912 - g 接收波束: 外层动作产出（每 UAV 2N 维实/虚部，L2 归一化后 ‖g_i‖²=1）
            "uav_rec_beam_dir_real": self.base_args.uavs_num * self.base_args.antenna_nums,
            "uav_rec_beam_dir_imag": self.base_args.uavs_num * self.base_args.antenna_nums,
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
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                -1.0,
                dtype=np.float32
            ),  # UAV 接收波束方向实部
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                -1.0,
                dtype=np.float32
            ),  # UAV 接收波束方向虚部
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
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                1.0,
                dtype=np.float32
            ),  # UAV 接收波束方向实部
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                1.0,
                dtype=np.float32
            ),  # UAV 接收波束方向虚部
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
        self.uavs_2_cus_channels, self.uavs_2_bs_channels, self.cus_2_bs_channels = compute_com_channel_gain(
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
        self.uavs_2_targets_channels = compute_sen_channel_gain(
            radar_rcs=self.base_args.radar_rcs,
            frac_d_lambda=self.base_args.frac_d_lambda,
            uavs_pos=self.cur_uavs_pos,
            targets_pos=self.init_targets_pos,
            antenna_nums=self.base_args.antenna_nums,
            ref_path_loss=db_2_watt(self.base_args.ref_path_loss_db)
        )

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
        num_windows = get_num_target_windows(
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
            flatten_complex(self.uavs_2_bs_channels),
            flatten_complex(self.uavs_2_cus_channels),
            flatten_complex(self.cus_2_bs_channels),
            self._get_all_target_coords(),
            self.cur_uavs_pos.astype(np.float32).flatten(),
            #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
            self._get_slots_until_switch(),
            self._get_current_uav_target_coords(),
            self._get_next_uav_target_coords(),
        ]).astype(np.float32)

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
        offset += self.base_args.cus_num

        # 20260912 - g 接收波束: 第 5/6 个动作头 -> 单位范数接收波束（外层给定量）
        rec_beam_dim = self.base_args.uavs_num * self.base_args.antenna_nums
        rec_beam_dir_real_flat = continuous_actions[offset:offset + rec_beam_dim]
        offset += rec_beam_dim
        rec_beam_dir_imag_flat = continuous_actions[offset:offset + rec_beam_dim]
        offset += rec_beam_dim
        uavs_rec_beam_vectors = build_unit_norm_rec_beam(
            rec_beam_dir_real_flat,
            rec_beam_dir_imag_flat,
            self.base_args.uavs_num,
            self.base_args.antenna_nums,
        )

        # NOTE - 将离散 CU 索引动作转换为 UAV-CU 匹配矩阵输入 CCCP，以适配 reward 计算接口
        uavs_cus_matched_matrix = build_uavs_cus_matched_matrix(
            discrete_actions,
            self.base_args.uavs_num,
            self.base_args.cus_num,
        )

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
                f"‖g‖={np.linalg.norm(uavs_rec_beam_vectors[i]):.4f}, "
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
            cus_entertaining_task_size=self.cus_entertaining_task_size,
            uavs_rec_beam_vectors=uavs_rec_beam_vectors,
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
        self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t],
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
        )

        next_state_dict = {"bs": self._build_bs_observation()}
        done = int(self.t >= self.madrl_args.total_time_slots)
        return next_state_dict, float(total_reward), reward, done, energy_opt, success

    def reset(self):
        self.t = 0
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t],
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
        )
        self._refresh_channels()
        return {"bs": self._build_bs_observation()}

    def getPosUAV(self):
        return self.cur_uavs_pos.copy()

    def getPosCU(self):
        return self.cur_cus_pos.copy()

    def getPosTarget(self):
        return self.init_targets_pos.copy()

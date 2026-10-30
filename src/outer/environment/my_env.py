try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
import numpy as np
import hashlib
import copy
import json
from datetime import datetime

# 训练时环境单步诊断由 learner 的 episode 聚合日志承载，避免多进程终端交错输出。
def _env_debug_print(*args, **kwargs):
    return None

print = _env_debug_print

from environment.channel_models import (
    compute_com_channel_gain,
    compute_sen_channel_gain,
    db_2_watt,
    dbm_2_watt,
)
from environment.mapping import (
    build_unit_norm_rec_beam,
    build_matched_rec_beam,
    build_mvdr_rec_beam,
    build_mvdr_residual_rec_beam,
    mix_reference_rec_beam,
    build_current_target_steering,
    build_phase_residual_rec_beam,
    receive_beam_mapping_metadata,
    build_uav_targets_matched_matrix,
    build_uavs_cus_matched_matrix,
)
from environment.observation import (
    ObsContext,
    BS_OBSERVATION_CONTRACT,
    REFERENCE_OBS_DIM,
    build_bs_observation,
    compute_bs_obs_dim,
    format_bs_observation_layout,
    is_reference_scene,
    validate_bs_observation_layout,
)
from configs.seed_manager import scenario_rng
from environment.cu_tasks import CUTaskSchedule
from environment.topology import (
    generate_cu_trajectory,
    generate_pos,
    generate_uav_target_schedule,
    get_num_target_windows,
    validate_target_schedule_requirements,
    resolve_global_ring_hold_slots,
    print_precomputed_target_schedule,
)
from reward.my_reward import MyReward
from reward.episode_success_bonus import validate_episode_success_bonus


class MyEnv(gym.Env):
    """外层 ISCC 场景环境（MHBPPO 单智能体 BS 视角）。

    职责边界：本类只保留环境状态机（时隙 `t`、位置状态）、动作/观测空间、`reset` /
    `step`，以及把当前状态组装成只读快照后的 delegate。与状态机无关的通用计算已抽离：
    - 信道与单位换算 → `environment.channel_models`
    - 位置/轨迹/目标调度 → `environment.topology`
    - 匹配矩阵/接收波束/扁平化 → `environment.mapping`
    - 观测分段清单与相对几何量编码 → `environment.observation`
    """

    def __init__(self, base_args, madrl_args, seed_offset=0):
        super(MyEnv, self).__init__()
        # Runtime solver controls belong to this environment's parameter copy;
        # constructing the learner's probe must not mutate shared Parameters.
        if hasattr(madrl_args, "mosek_num_threads"):
            self.base_args = copy.deepcopy(base_args)
            solver_threads = int(getattr(madrl_args, "mosek_num_threads", 0))
            self.base_args.mosek_num_threads = solver_threads
            params = getattr(self.base_args, "_params", None)
            if params is not None:
                params.mosek_num_threads = solver_threads
        else:
            self.base_args = (copy.deepcopy(base_args)
                              if getattr(base_args, "target_layout", "random_sector") == "global_ring"
                              else base_args)
        self.madrl_args = madrl_args
        self.receive_beam_mapping = receive_beam_mapping_metadata(
            getattr(madrl_args, "receive_beam_phase_mode", "absolute"),
            getattr(madrl_args, "receive_beam_phase_limit_deg", 45.0),
            getattr(madrl_args, "structured_beam_alpha", 1.0),
            getattr(madrl_args, "mvdr_residual", False),
            getattr(madrl_args, "mvdr_residual_radius", 0.2),
            getattr(madrl_args, "structured_beam_reference", "mvdr"))
        self.mvdr_residual_enabled = bool(getattr(madrl_args, "mvdr_residual", False))
        self.phase_residual_enabled = self.receive_beam_mapping["receive_beam_phase_mode"] == "phase_residual"
        phase_limit = np.deg2rad(self.receive_beam_mapping["receive_beam_phase_limit_deg"] or 45.0)
        self.epsilon = 1e-4
        self.t = 0
        self.last_action_trace = None  # 最近一步动作的只读日志快照（不改变 step 返回契约）
        self.last_env_diagnostic = None
        self.env_diagnostics = bool(getattr(madrl_args, "env_diagnostics", True))
        self.episode_success_bonus = validate_episode_success_bonus(
            getattr(madrl_args, "episode_success_bonus", 0.0))
        if self.episode_success_bonus != 0.0:
            raise ValueError("episode success bonus was removed; coefficient must be 0")
        self._reset_episode_success_tracking()
        self.diagnostic_candidate_interval = max(1, int(getattr(madrl_args, "diagnostic_candidate_interval", 1)))
        self.diagnostic_snapshot_limit = max(0, int(getattr(madrl_args, "diagnostic_snapshot_limit", 1)))
        self._diagnostic_snapshots_used = 0
        self._diagnostic_scene = None
        self.target_layout = getattr(self.base_args, "target_layout", "random_sector")
        hold_slots = getattr(self.base_args, "target_hold_slots", 4)
        if self.target_layout == "global_ring":
            hold_slots = resolve_global_ring_hold_slots(
                self.base_args.uavs_num, self.base_args.targets_num, madrl_args.total_time_slots)
            # Keep the physical alias, effective configuration and schedule consistent.
            self.base_args.target_hold_slots = hold_slots
        if isinstance(hold_slots, bool) or int(hold_slots) != hold_slots or hold_slots <= 0:
            raise ValueError("target_hold_slots must be a positive integer")
        self.target_hold_slots = int(hold_slots)
        # ── 场景随机化配置（并行 rollout 状态多样性） ─────────────────────────
        # Scene identity is independent of the worker's action RNG stream.
        # Shared-scene synchronous workers use the same scene offset (default 0);
        # legacy asynchronous workers may still use distinct scene offsets.
        self.seed_offset = int(seed_offset)
        self.randomize_layout_per_episode = bool(
            getattr(madrl_args, "randomize_layout_per_episode", False))
        self.randomize_cu_traj_per_episode = bool(
            getattr(madrl_args, "randomize_cu_traj_per_episode", False))
        self.randomize_nlos_per_episode = bool(
            getattr(madrl_args, "randomize_nlos_per_episode", False))
        self.debug_topology = bool(getattr(madrl_args, "debug_topology", False))
        self._episode_idx = 0  # 当前场景对应的 episode 序号（reset 时推进）
        # Independent task RNG: no episode or action RNG state participates.
        self.cu_tasks = CUTaskSchedule(self.base_args, madrl_args, self.seed_offset)
        self._select_current_tasks()
        # 生成初始 UAV / CU / 目标位置，并预计算 CU 轨迹和 UAV-目标分配调度
        self._generate_scenario(scenario_rng(self.base_args.seed, self.seed_offset, 0))

        self.bs_continuous_action_splits = {
            "uav_angles": self.base_args.uavs_num,
            "uav_distances": self.base_args.uavs_num,
            "uav_off_durations": self.base_args.uavs_num,
            "cu_off_powers": self.base_args.cus_num,
            # g 接收波束：每 UAV N 维幅度 + N 维相位，L2 归一化后 ‖g_i‖²=1
            "uav_rec_beam_amplitude": self.base_args.uavs_num * self.base_args.antenna_nums,
            "uav_rec_beam_phase": self.base_args.uavs_num * self.base_args.antenna_nums,
        }
        # NOTE - UAV 个离散头: 每个 UAV 选择一个 CU 索引进行匹配
        self.bs_discrete_action_dims = np.full(self.base_args.uavs_num, self.base_args.cus_num, dtype=np.int64)

        # 与 inner/outer_sampler 的可行样本范围对齐：
        # CU 功率避免接近 0；UAV 卸载时长为 [0.01, 0.09] s，
        # 为 D_sensing + D_off < D_sensing_max 留出两侧安全裕量。
        cu_max_power = dbm_2_watt(self.base_args.cu_max_power_dbm)
        uav_off_duration_low = 0.01
        uav_off_duration_high = 0.09
        bs_continuous_low = np.concatenate([
            np.zeros(self.base_args.uavs_num, dtype=np.float32),  # UAV 飞行角度
            np.full(
                self.base_args.uavs_num,
                self.base_args.uav_min_speed * float(self.cu_tasks.slot_seconds.min()),
                dtype=np.float32
            ), # UAV 飞行距离
            np.full(self.base_args.uavs_num, uav_off_duration_low, dtype=np.float32),  # UAV 卸载时长
            np.full(self.base_args.cus_num, 0.1 * cu_max_power, dtype=np.float32),  # CU 卸载功率
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                0.0,
                dtype=np.float32
            ),  # UAV 接收波束幅度
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                0.0,
                dtype=np.float32
            ),  # UAV 接收波束相位（rad）
        ])
        bs_continuous_high = np.concatenate([
            np.full(self.base_args.uavs_num, 2 * np.pi, dtype=np.float32),   # UAV 飞行角度
            np.full(
                self.base_args.uavs_num,
                self.base_args.uav_max_speed * float(self.cu_tasks.slot_seconds.max()),
                dtype=np.float32
            ),  # UAV 飞行距离
            np.full(self.base_args.uavs_num, uav_off_duration_high, dtype=np.float32),  # UAV 卸载时长
            np.full(self.base_args.cus_num, cu_max_power, dtype=np.float32),  # CU 卸载功率
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                1.0,
                dtype=np.float32
            ),  # UAV 接收波束幅度
            np.full(
                self.base_args.uavs_num * self.base_args.antenna_nums,
                2 * np.pi,
                dtype=np.float32
            ),  # UAV 接收波束相位（rad）
        ])

        if self.phase_residual_enabled:
            beam_dim = self.base_args.uavs_num * self.base_args.antenna_nums
            bs_continuous_low[-beam_dim:] = -phase_limit
            bs_continuous_high[-beam_dim:] = phase_limit
        elif self.mvdr_residual_enabled:
            beam_dim = self.base_args.uavs_num * self.base_args.antenna_nums
            bs_continuous_high[-beam_dim:] = 1.0  # imaginary latent, not radians

        self.action_space = {
            "bs": {
                "continuous": spaces.Box(low=bs_continuous_low, high=bs_continuous_high, dtype=np.float32),
                "discrete": spaces.MultiDiscrete(self.bs_discrete_action_dims),
            }
        }

        # BS 观测：信道、按调度顺序的三维相对目标、当前目标标志、倒计时与任务。
        obs_dim_bs = compute_bs_obs_dim(self.base_args)
        self.observation_space = {
            "bs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_bs,), dtype=np.float32)
        }

        self.reward_calculator = MyReward(
            self.base_args, enable_diagnostics=self.env_diagnostics,
            sensing_penalty_coef=getattr(madrl_args, "sensing_penalty_coef", 0.0),
            sensing_penalty_reference_factor=getattr(
                madrl_args, "sensing_penalty_reference_factor", 10.0),
        )
        self._validate_bs_observation_layout()

    def _select_current_tasks(self):
        (self.cus_entertaining_task_size, self.cus_task_cycles,
         self.cus_task_deadlines, self.current_slot_duration) = self.cu_tasks.row(self.t)

    def get_current_action_bounds(self):
        """Box is the episode envelope; the actual distance mapping uses this slot's tau."""
        low = self.action_space["bs"]["continuous"].low.copy()
        high = self.action_space["bs"]["continuous"].high.copy()
        section = slice(self.base_args.uavs_num, 2 * self.base_args.uavs_num)
        low[section] = self.base_args.uav_min_speed * self.current_slot_duration
        high[section] = self.base_args.uav_max_speed * self.current_slot_duration
        return low, high

    def _current_task_record(self):
        return {"L_bits": self.cus_entertaining_task_size.tolist(),
                "C_cycles_per_bit": self.cus_task_cycles.tolist(),
                "D_seconds": self.cus_task_deadlines.tolist(),
                "slot_seconds": self.current_slot_duration,
                "elapsed_seconds": float(self.cu_tasks.time_boundaries[self.t]),
                "slot_end_seconds": float(self.cu_tasks.time_boundaries[self.t + 1])}

    def _generate_scenario(self, rng):
        """按给定随机源生成整套场景：布点、CU 轨迹、NLoS 分量、扇区目标调度。

        `__init__` 以 `scenario_rng(seed, seed_offset, 0)` 调用一次生成初始场景；
        `reset(episode_idx)` 在开启对应随机化开关时以新派生的 episode rng 重跑本方法，
        实现「跨 worker 互异 + 跨 episode 互异 + 同参数可复现」的状态多样性。
        """
        # 场景重生成从 t=0 开始（内部多处按 self.t 取预生成量）
        self.t = 0
        self._diagnostic_scene = None
        validate_target_schedule_requirements(
            self.base_args.uavs_num, self.base_args.targets_num,
            self.madrl_args.total_time_slots, self.target_hold_slots)
        # ── 布点（扇区分区部署，显式 rng） ─────────────────────────────────
        (self.init_uavs_pos, self.init_cus_pos, self.init_targets_pos,
         self.target_sector, self.target_layout_info) = generate_pos(
            self.base_args.uavs_num,
            self.base_args.cus_num,
            self.base_args.targets_num,
            self.base_args.center,
            self.base_args.radius,
            self.base_args.uav_height,
            rng=rng,
            target_layout=self.target_layout,
            target_hold_slots=self.target_hold_slots,
            time_slot_duration=self.base_args.time_slot_duration,
            uav_min_speed=self.base_args.uav_min_speed,
            uav_max_speed=self.base_args.uav_max_speed,
            target_ring_space_ratio=getattr(self.base_args, "target_ring_space_ratio", 1.0 / 3.0),
            target_ring_move_ratio=getattr(self.base_args, "target_ring_move_ratio", 0.85),
            target_ring_direction=getattr(self.base_args, "target_ring_direction", "clockwise"),
            target_ring_radius_ratio=getattr(self.base_args, "target_ring_radius_ratio", 0.5),
            uav_safe_distance=self.base_args.uav_safe_distance,
            return_layout_info=True,
        )
        if self.target_layout == "global_ring":
            self.target_layout_info.update({
                "hold_slots": self.target_hold_slots,
                "total_time_slots": int(self.madrl_args.total_time_slots),
                "window_rule": "total_time_slots / (targets_num / uavs_num)",
            })
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.uav_previous_displacement = np.zeros_like(self.cur_uavs_pos)

        # ── CU 马尔可夫轨迹（同一 rng 流，与本 episode 布点同源） ───────────
        self.precomputed_cus_traj, self.precomputed_cus_velocities = generate_cu_trajectory(
            init_cus_pos=self.init_cus_pos,
            cus_num=self.base_args.cus_num,
            total_time_slots=self.madrl_args.total_time_slots,
            markov_velocity=self.base_args.markov_velocity,
            markov_memory_level=self.base_args.markov_memory_level,
            markov_asymptotic_mean_of_velocity=self.base_args.markov_asymptotic_mean_of_velocity,
            markov_standard_deviation_of_velocity=self.base_args.markov_standard_deviation_of_velocity,
            time_slot_duration=self.cu_tasks.slot_seconds,
            seed=rng,
            return_velocities=True,
        )
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()

        # ── NLoS 分量：按时隙预生成通信链路的高斯散射样本 ────────────────────
        self.precomputed_nlos_components = self._precompute_nlos_components(rng)

        # ── 扇区内链式目标调度（确定性，依赖布点） ─────────────────────────
        self.precomputed_uav_target_schedule, self.precomputed_uav_target_schedule_distances = generate_uav_target_schedule(
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
            init_uavs_pos=self.init_uavs_pos,
            init_targets_pos=self.init_targets_pos,
            total_time_slots=self.madrl_args.total_time_slots,
            hold_slots=self.target_hold_slots,
            center=self.base_args.center,
            target_sector=self.target_sector,
            target_layout=self.target_layout,
        )
        # 调度表仅在 debug 模式打印（正常训练逐 episode 重生成场景，打印会刷屏）
        if self.debug_topology:
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

    def _precompute_nlos_components(self, rng=None):
        # NLoS 分量: 按时隙预生成通信链路的 NLoS 高斯样本。
        # rng 为 None 时回退基线 seed（兼容旧调用）；逐 episode 重生成时传入
        # 已派生的独立 Generator，保证跨 worker / 跨 episode 互异。
        total_slots = self.madrl_args.total_time_slots + 1
        rng = np.random.default_rng(self.base_args.seed) if rng is None else rng
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
            nlos_components=slot_nlos,
            bs_pos=self.base_args.bs_pos,
        )
        self.uavs_2_targets_channels = compute_sen_channel_gain(
            radar_rcs=self.base_args.radar_rcs,
            frac_d_lambda=self.base_args.frac_d_lambda,
            uavs_pos=self.cur_uavs_pos,
            targets_pos=self.init_targets_pos,
            antenna_nums=self.base_args.antenna_nums,
            ref_path_loss=db_2_watt(self.base_args.ref_path_loss_db)
        )

    def _get_slots_until_switch(self):
        #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
        effective_t = min(self.t, max(self.madrl_args.total_time_slots - 1, 0))
        slots_until_switch = self.target_hold_slots - 1 - (effective_t % self.target_hold_slots)
        return np.array([slots_until_switch], dtype=np.float32)

    def _get_current_uav_target_indices(self):
        # 20260916 - 观测相对化(L2): 只返回目标索引，坐标编码交给 environment.observation
        effective_t = min(self.t, self.madrl_args.total_time_slots)
        return self.precomputed_uav_target_schedule[effective_t]

    def _get_next_uav_target_indices(self):
        #20260408 - 观测空间修改 : 加入下一个窗口感知目标和切换倒计时
        # 20260916 - 观测相对化(L2): 越界保护保留，返回值由「目标坐标」改为「目标索引」
        if self.madrl_args.total_time_slots <= 0:
            return self._get_current_uav_target_indices()
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
        return self.precomputed_uav_target_schedule[next_window_start_slot]

    def _build_obs_context(self):
        """汇总当前时刻的只读快照，供 `environment.observation` 的纯函数消费。"""
        return ObsContext(
            uavs_2_bs_channels=self.uavs_2_bs_channels,
            uavs_2_cus_channels=self.uavs_2_cus_channels,
            cus_2_bs_channels=self.cus_2_bs_channels,
            uavs_pos=self.cur_uavs_pos,
            uav_previous_displacement=self.uav_previous_displacement,
            cus_velocities=self.precomputed_cus_velocities[self.t],
            targets_pos=self.init_targets_pos,
            target_sequence_indices=self.precomputed_uav_target_schedule[
                0:self.madrl_args.total_time_slots:self.target_hold_slots].T,
            current_target_window=min(self.t // self.target_hold_slots,
                                      self.madrl_args.total_time_slots // self.target_hold_slots - 1),
            slots_until_switch=self._get_slots_until_switch(),
            task_sizes=np.asarray(self.cus_entertaining_task_size, dtype=np.float32),
            task_cycles=np.asarray(self.cus_task_cycles, dtype=np.float32),
            task_delays=np.asarray(self.cus_task_deadlines, dtype=np.float32),
        )

    def _build_bs_observation(self):
        return build_bs_observation(self._build_obs_context())

    def _validate_bs_observation_layout(self):
        """启动自检：分段维度 == 实际拼接长度，且总维度与 `observation_space` 一致。

        20260916 - 观测相对化(L2): 清单式单一数据源已消除顺序错位，此处再兜住
        「单段声明维度与产出长度不一致」这类手误，避免错误维度被静默喂给策略网络。
        """
        print(format_bs_observation_layout(self.base_args))
        ctx = self._build_obs_context()
        mismatched = [
            (name, declared, built)
            for name, declared, built in validate_bs_observation_layout(self.base_args, ctx)
            if declared != built
        ]
        assert not mismatched, "BS 观测分段维度与拼接长度不一致: {}".format(mismatched)

        obs = build_bs_observation(ctx)
        declared_total = int(self.observation_space["bs"].shape[0])
        assert obs.shape[0] == declared_total, (
            "BS 观测维度错位: observation_space={} vs 实际拼接={}".format(
                declared_total, obs.shape[0]
            )
        )
        assert np.all(np.isfinite(obs)), "BS 观测包含 NaN/Inf（检查信道与目标位置计算）"
        if is_reference_scene(self.base_args):
            assert obs.shape[0] == REFERENCE_OBS_DIM, (
                "默认场景 (I=4, J=10, K=40, N=10) 的 BS 观测应为 {} 维，实测 {} 维".format(
                    REFERENCE_OBS_DIM, obs.shape[0]
                )
            )
        print(
            "[BS Observation Layout] 自检通过: dim={}, 取值范围=[{:.4f}, {:.4f}]".format(
                obs.shape[0], float(obs.min()), float(obs.max())
            )
        )

    def _exogenous_component_hashes(self):
        """Fingerprint the full exogenous scene, including the immutable task table."""
        digest = hashlib.sha256()
        components = {"uav_initial": self.init_uavs_pos, "cu_initial": self.init_cus_pos,
                      "targets": self.init_targets_pos, "cu_trajectory": self.precomputed_cus_traj,
                      "cu_velocities": self.precomputed_cus_velocities,
                      "target_schedule": self.precomputed_uav_target_schedule,
                      "cu_task_L": self.cu_tasks.l_bits, "cu_task_C": self.cu_tasks.c_cycles,
                      "cu_task_D": self.cu_tasks.d_seconds,
                      "slot_durations": self.cu_tasks.slot_seconds,
                      **self.precomputed_nlos_components}
        hashes = {}
        for name, values in components.items():
            values = np.ascontiguousarray(values)
            fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
            hashes[name] = fingerprint
            digest.update((name + str(values.shape) + str(values.dtype) + fingerprint).encode())
        return digest.hexdigest(), hashes

    def get_exogenous_scene_signature(self):
        """Identify fixed external time series and physical/task constants, even with diagnostics off.

        UAV positions affected by actions and resulting geometric channels are deliberately
        absent: equal external scenes need not have equal controlled trajectories.
        """
        scene_id, hashes = self._exogenous_component_hashes()
        physical_keys = (
            "uavs_num", "cus_num", "targets_num", "antenna_nums", "area_length", "radius",
            "center", "bs_pos", "uav_height", "uav_min_speed", "uav_max_speed",
            "uav_safe_distance", "uav_c1", "uav_c2", "bandwidth", "noise_power_density_dbm",
            "ref_path_loss_db", "rician_factor_db", "alpha_uav_link", "alpha_cu_link", "alpha_2",
            "frac_d_lambda", "radar_rcs", "sen_sinr", "radar_duty_ratio", "var_range_fluctuation",
            "radar_impulse_duration", "radar_spectrum_shape", "uav_sen_duration",
            "bs_cycles_per_bit", "bs_max_freq", "kappa", "cu_max_power_dbm", "uav_max_power",
            "cu_max_delay", "uav_max_delay", "time_slot_duration", "omega_weight_1",
            "omega_weight_2", "omega_weight_3", "L_cu_task_bits", "target_layout",
            "target_ring_space_ratio", "target_ring_move_ratio", "target_ring_radius_ratio", "markov_velocity",
            "markov_memory_level", "markov_asymptotic_mean_of_velocity",
            "markov_standard_deviation_of_velocity",
        )
        constants = {name: getattr(self.base_args, name) for name in physical_keys
                     if hasattr(self.base_args, name)}
        constants.update({"total_time_slots": int(self.madrl_args.total_time_slots),
                          "communication_channel_contract": "configured_bs_position_v2",
                          "scene_fingerprint_contract": "exogenous_with_cu_velocity_v2",
                          "observation_contract": BS_OBSERVATION_CONTRACT,
                          "target_hold_slots": self.target_hold_slots,
                          "target_layout_info": self.target_layout_info,
                          "target_sector": self.target_sector,
                          "cu_task_settings": self.cu_tasks.settings})
        def serialize(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"unsupported physical constant: {type(value).__name__}")
        constant_sha = hashlib.sha256(json.dumps(constants, sort_keys=True, separators=(",", ":"),
            default=serialize, allow_nan=False).encode()).hexdigest()
        regenerated = (self.randomize_layout_per_episode or
                       self.randomize_cu_traj_per_episode or self.randomize_nlos_per_episode)
        return {"signature": hashlib.sha256((scene_id + constant_sha).encode()).hexdigest(),
                "scene_fingerprint_contract": "exogenous_with_cu_velocity_v2",
                "scene_id": scene_id, "component_sha256": hashes, "constant_sha256": constant_sha,
                "master_seed": int(self.base_args.seed), "scene_seed_offset": self.seed_offset,
                "scenario_episode_idx": int(self._episode_idx if regenerated else 0)}

    def get_diagnostic_scene(self):
        """Scenario record cached per generated scene; learner writes it once."""
        if not self.env_diagnostics:
            return None
        if self._diagnostic_scene is None:
            scene_id, hashes = self._exogenous_component_hashes()
            regenerated = (self.randomize_layout_per_episode or
                           self.randomize_cu_traj_per_episode or self.randomize_nlos_per_episode)
            scenario_episode = self._episode_idx if regenerated else 0
            self._diagnostic_scene = {
                "scene_id": scene_id, "base_seed": int(self.base_args.seed),
                "seed_offset": self.seed_offset, "scenario_episode_idx": int(scenario_episode),
                "rng_entropy": int(self.base_args.seed) + self.seed_offset,
                "rng_spawn_key": [int(scenario_episode)], "rng_bit_generator": "PCG64",
                "randomization": {"layout": self.randomize_layout_per_episode,
                                  "cu_trajectory": self.randomize_cu_traj_per_episode,
                                  "nlos": self.randomize_nlos_per_episode},
                "regeneration_semantics": "any_enabled_flag_regenerates_all_components",
                "target_hold_slots": self.target_hold_slots,
                "bs_position": list(self.base_args.bs_pos),
                "communication_channel_contract": "configured_bs_position_v2",
                 "scene_fingerprint_contract": "exogenous_with_cu_velocity_v2",
                "observation_contract": BS_OBSERVATION_CONTRACT,
                "target_layout": self.target_layout_info,
                "initial_uav_positions": self.init_uavs_pos.tolist(),
                "initial_cu_positions": self.init_cus_pos.tolist(),
                "target_positions": self.init_targets_pos.tolist(),
                "target_schedule": self.precomputed_uav_target_schedule.tolist(),
                "cu_tasks": self.cu_tasks.metadata(),
                "component_sha256": hashes,
            }
        return self._diagnostic_scene

    def step(self, actions, i_episode=None):
        if self.t >= self.madrl_args.total_time_slots:
            raise RuntimeError("episode has terminated; reset before stepping again")
        self.last_action_trace = None
        self.last_env_diagnostic = None
        bs_actions = actions["bs"] if isinstance(actions, dict) and "bs" in actions else actions

        continuous_actions = np.asarray(bs_actions["continuous"], dtype=np.float32)
        discrete_actions = np.asarray(bs_actions["discrete"], dtype=np.int64)

        continuous_low, continuous_high = self.get_current_action_bounds()
        clipped_continuous_actions = np.clip(continuous_actions, 0.0, 1.0)
        continuous_actions = clipped_continuous_actions * (continuous_high - continuous_low) + continuous_low

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
        rec_beam_amplitude_flat = continuous_actions[offset:offset + rec_beam_dim]
        offset += rec_beam_dim
        rec_beam_phase_flat = continuous_actions[offset:offset + rec_beam_dim]
        offset += rec_beam_dim
        reference_beams = build_current_target_steering(
            self.uavs_2_targets_channels, self._get_current_uav_target_indices())
        if self.phase_residual_enabled:
            uavs_rec_beam_vectors = build_phase_residual_rec_beam(
                rec_beam_amplitude_flat, rec_beam_phase_flat, reference_beams)
        else:
            uavs_rec_beam_vectors = build_unit_norm_rec_beam(
                rec_beam_amplitude_flat, rec_beam_phase_flat,
                self.base_args.uavs_num, self.base_args.antenna_nums)
        normalized_rec_beam_vectors = uavs_rec_beam_vectors.copy()
        mvdr_beams = None
        # MVDR is conditioned on this slot's actual matching and CU power.
        uavs_cus_matched_matrix = build_uavs_cus_matched_matrix(
            discrete_actions, self.base_args.uavs_num, self.base_args.cus_num)
        # Structured receive-beam reference.  The actor still samples the
        # latent direction, so its PPO log-probability remains well-defined;
        # this deterministic transform only changes the physical beam sent to
        # the inner solver.
        beam_alpha = float(np.clip(
            getattr(self.madrl_args, "structured_beam_alpha", 1.0), 0.0, 1.0
        ))
        matched_reference = getattr(self.madrl_args, "structured_beam_reference", "mvdr") == "matched"
        if not matched_reference and (self.mvdr_residual_enabled or beam_alpha < 1.0):
            mvdr_beams = build_mvdr_rec_beam(
                self.uavs_2_targets_channels, self._get_current_uav_target_indices(),
                self.uavs_2_cus_channels, uavs_cus_matched_matrix, cus_off_power,
                dbm_2_watt(self.base_args.noise_power_density_dbm) * self.base_args.bandwidth,
            )
        if matched_reference:
            uavs_rec_beam_vectors = build_matched_rec_beam(
                self.uavs_2_targets_channels, self._get_current_uav_target_indices())
        elif self.mvdr_residual_enabled:
            uavs_rec_beam_vectors = build_mvdr_residual_rec_beam(
                mvdr_beams, clipped_continuous_actions[offset - 2 * rec_beam_dim:offset - rec_beam_dim],
                clipped_continuous_actions[offset - rec_beam_dim:offset],
                getattr(self.madrl_args, "mvdr_residual_radius", 0.2))
        elif beam_alpha == 0.0:
            uavs_rec_beam_vectors = mvdr_beams
        elif beam_alpha < 1.0:
            uavs_rec_beam_vectors = mix_reference_rec_beam(mvdr_beams, uavs_rec_beam_vectors, beam_alpha)

        # Actual normalized target gain; residual fields are radians, not azimuth.
        beam_diagnostics = {
            **self.receive_beam_mapping,
            "target_direction_gain": np.abs(np.sum(
                uavs_rec_beam_vectors.conj() * reference_beams, axis=1)) ** 2,
        }
        beam_diagnostics["target_direction_gain"] = beam_diagnostics["target_direction_gain"].tolist()
        if mvdr_beams is not None:
            beam_diagnostics["mvdr_direction_similarity"] = np.abs(np.sum(
                mvdr_beams.conj() * uavs_rec_beam_vectors, axis=1)).tolist()
        if self.phase_residual_enabled:
            limit = np.deg2rad(self.receive_beam_mapping["receive_beam_phase_limit_deg"])
            beam_diagnostics.update(
                phase_residual_rad=rec_beam_phase_flat.reshape(reference_beams.shape).tolist(),
                phase_residual_boundary_fraction=float(np.mean(np.abs(rec_beam_phase_flat) >= 0.95 * limit)),
                zero_amplitude_fallback=(np.linalg.norm(
                    rec_beam_amplitude_flat.reshape(reference_beams.shape) * reference_beams,
                    axis=1) <= 1e-12).tolist())

        next_uavs_pos = self.cur_uavs_pos + np.stack([
            diff_distance * np.cos(diff_theta),
            diff_distance * np.sin(diff_theta),
            np.zeros_like(diff_distance),
        ], axis=1)

        # Snapshot the exact transformed values used below, before solver/state mutation.
        # Store complex beams as real/imaginary matrices for portable JSON logging.
        self.last_action_trace = {
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "time_slot": int(self.t),
            "cu_tasks": self._current_task_record(),
            "continuous_low": continuous_low.tolist(),
            "continuous_high": continuous_high.tolist(),
            "clipped_action": {
                "continuous": clipped_continuous_actions.tolist(),
                "discrete": np.argmax(uavs_cus_matched_matrix, axis=1).tolist(),
            },
            "mapped_continuous": continuous_actions.tolist(),
            "physical_action": {
                "uav_angles": diff_theta.tolist(),
                "uav_distances": diff_distance.tolist(),
                "uav_off_durations": off_duration.tolist(),
                "cu_off_powers": cus_off_power.tolist(),
                "uavs_cus_matched_matrix": uavs_cus_matched_matrix.tolist(),
                "uavs_targets_matched_matrix": self.uavs_targets_matched_matrix.tolist(),
                "uavs_pos_before": self.cur_uavs_pos.tolist(),
                "uavs_pos_after": next_uavs_pos.tolist(),
                "uavs_rec_beam_vectors": {
                    "real": uavs_rec_beam_vectors.real.tolist(),
                    "imag": uavs_rec_beam_vectors.imag.tolist(),
                },
            },
            "normalized_rec_beam_vectors": None if self.mvdr_residual_enabled else {
                "real": normalized_rec_beam_vectors.real.tolist(),
                "imag": normalized_rec_beam_vectors.imag.tolist(),
            },
            "structured_beam_alpha": beam_alpha,
            "receive_beam_diagnostics": beam_diagnostics,
            "mvdr_rec_beam_vectors": None if mvdr_beams is None else {
                "real": mvdr_beams.real.tolist(),
                "imag": mvdr_beams.imag.tolist(),
            },
        }

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
            cu_task_cycles=self.cus_task_cycles,
            cu_task_deadlines=self.cus_task_deadlines,
            slot_duration=self.current_slot_duration,
            uavs_rec_beam_vectors=uavs_rec_beam_vectors,
            diagnostic_candidates=(self.t % self.diagnostic_candidate_interval == 0),
            diagnostic_cu_min_power=float(self.action_space["bs"]["continuous"].low[3 * self.base_args.uavs_num]),
            capture_failed_solution=(getattr(self.madrl_args, "log_detail_level", "compact") == "full"
                                     and self._diagnostic_snapshots_used < self.diagnostic_snapshot_limit),
        )
        reward_components = reward.get("components", {})
        success = int(
            reward_components.get("no_solution_penalty", 0) == 0 and
            reward_components.get("uav_collision_penalty_sum", 0) == 0 and
            reward_components.get("bs_alloc_spectrum_penalty", 0) == 0
        )
        is_terminal_step = self.t + 1 >= self.madrl_args.total_time_slots
        # Preserve success history as an observation/log statistic, but no
        # terminal reward is added. Compatibility fields explicitly mark this.
        self._episode_all_success = self._episode_all_success and bool(success)
        reward_components = dict(reward_components)
        reward_components.update({
            "slot_success": success,
            "episode_all_success": self._episode_all_success,
            "slot_reward_before_episode_bonus": float(total_reward),
            "episode_success_bonus": 0.0,
            "episode_success_bonus_applied": 0.0,
            "episode_success_bonus_contract": "removed_v1",
        })
        reward["components"] = reward_components
        reward["bs"] = float(total_reward)
        if self.env_diagnostics:
            diagnostic = reward.get("diagnostics")
            if diagnostic is None:
                raise RuntimeError("内层求解未返回已启用的环境诊断")
            if "problem_snapshot" in diagnostic:
                self._diagnostic_snapshots_used += 1
            self.last_env_diagnostic = {
                **diagnostic, "time_slot": int(self.t),
                "cu_tasks": self._current_task_record(),
                "scene_id": self.get_diagnostic_scene()["scene_id"],
                "target_indices": self._get_current_uav_target_indices().tolist(),
                "structured_beam_alpha": beam_alpha,
                "receive_beam_diagnostics": beam_diagnostics,
                "reward_components": reward.get("components", {}),
                "legacy_reward": float(reward_components["slot_reward_before_episode_bonus"]),
                "legacy_energy": float(energy_opt),
                "base_reward": float(reward_components.get(
                    "base_reward", reward_components["slot_reward_before_episode_bonus"])),
                "actual_reward": float(total_reward),
                "legacy_success": success,
            }
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
        print(f"  {color_penalty}uav_collision_penalty_sum{color_reset}: {reward_components.get('uav_collision_penalty_sum', 0.0):.4f}")
        print(f"  {color_penalty}no_solution_penalty{color_reset}: {reward_components.get('no_solution_penalty', 0.0):.4f}")
        print(
            f"=================================================="
        )
        self.t += 1
        self._select_current_tasks()
        self.uav_previous_displacement = next_uavs_pos - self.cur_uavs_pos
        self.cur_uavs_pos = next_uavs_pos
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self._refresh_channels()
        self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t],
            uavs_num=self.base_args.uavs_num,
            targets_num=self.base_args.targets_num,
        )

        next_state_dict = {"bs": self._build_bs_observation()}
        done = int(is_terminal_step)
        return next_state_dict, float(total_reward), reward, done, energy_opt, success

    def reset(self, episode_idx=None):
        """复位环境；开启随机化开关时按 episode_idx 重生成整套场景。

        :param episode_idx: episode 序号（0 起）。None 时使用内部计数器自增，
            episode 索引始终推进；仅随机化开启时才重生成场景。种子经
            `scenario_rng(base_seed, seed_offset, episode_idx)` 确定性派生：
            同 (seed, seed_offset, episode_idx) 场景严格可复现，不同
            worker / 不同 episode 互异。
        """
        self.last_action_trace = None
        self.last_env_diagnostic = None
        self._diagnostic_snapshots_used = 0
        self._reset_episode_success_tracking()
        if episode_idx is None:
            episode_idx = self._episode_idx + 1
        self._episode_idx = int(episode_idx)
        regenerate = (self.randomize_layout_per_episode
                      or self.randomize_cu_traj_per_episode
                      or self.randomize_nlos_per_episode)
        if regenerate:
            # 逐 episode 场景重生成：布点 / CU 轨迹 / NLoS / 调度全部换新
            self._generate_scenario(
                scenario_rng(self.base_args.seed, self.seed_offset, episode_idx))
        else:
            self.t = 0
            self.cur_uavs_pos = self.init_uavs_pos.copy()
            self.uav_previous_displacement = np.zeros_like(self.cur_uavs_pos)
            self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
            self.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
                self.precomputed_uav_target_schedule[self.t],
                uavs_num=self.base_args.uavs_num,
                targets_num=self.base_args.targets_num,
            )
            self._refresh_channels()
        self._select_current_tasks()
        return {"bs": self._build_bs_observation()}

    def _reset_episode_success_tracking(self):
        """Start a fresh all-slots-success accumulator for one episode."""
        self._episode_all_success = True

    def getPosUAV(self):
        return self.cur_uavs_pos.copy()

    def getPosCU(self):
        return self.cur_cus_pos.copy()

    def getPosTarget(self):
        return self.init_targets_pos.copy()

"""
pure-my-env.py
~~~~~~~~~~~~~~
纯端到端 PPO 环境：继承 MyEnv 以复用观测空间/信道/位置生成等逻辑，
将动作空间扩展为全部 P0 优化变量，step() 直接调用 PureMyReward 完成
能耗计算与约束检验，不依赖 CCCP 内层优化。
"""

import os
import sys
import importlib.util
import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

# ── 动态加载 environment/my_env 模块（避免 IDE "Unresolved reference"） ──
_my_env_path = os.path.join(_parent_dir, "environment", "my_env.py")
_spec_my_env = importlib.util.spec_from_file_location("_my_env", _my_env_path)
_mod_my_env = importlib.util.module_from_spec(_spec_my_env)
_spec_my_env.loader.exec_module(_mod_my_env)
MyEnv = _mod_my_env.MyEnv
dbm_2_watt = _mod_my_env.dbm_2_watt
db_2_watt = _mod_my_env.db_2_watt

# ── 动态加载 reward/pure-my-reward 模块 ──
_reward_path = os.path.join(_parent_dir, "reward", "pure-my-reward.py")
_spec_reward = importlib.util.spec_from_file_location("_pure_my_reward", _reward_path)
_mod_reward = importlib.util.module_from_spec(_spec_reward)
_spec_reward.loader.exec_module(_mod_reward)
PureMyReward = _mod_reward.PureMyReward


class PureMyEnv(MyEnv):
    """
    纯端到端 PPO 环境。

    与父类 MyEnv 的区别：
    - 动作空间包含全部 P0 优化变量（波束向量、频率、CU 时长等）
    - step() 不再调用 CCCP，直接由 PureMyReward 计算 reward
    - 观测空间完全一致（继承自 MyEnv）
    """

    def __init__(self, base_args, madrl_args):
        # ── 先调父类构造（全部信道/位置/目标逻辑一致） ──
        super().__init__(base_args, madrl_args)

        # ── 覆盖 reward_calculator ──
        self.reward_calculator = PureMyReward(base_args)

        # ── 扩展动作空间 ──
        self._build_pure_action_space()

        # ── 测试层 feasibility enforcement 开关 ──
        self.feasibility_enforce = False

        # ── 修正观测空间：_build_bs_observation() 比父类多 1 维 sen_sinr ──
        obs_dim = self._build_bs_observation().shape[0]
        self.observation_space["bs"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _build_pure_action_space(self):
        """构造包含全部 P0 优化变量的动作空间。"""
        a = self.base_args
        I = a.uavs_num
        J = a.cus_num
        N = a.antenna_nums

        epsilon = 1e-4
        P_max_uav = a.uav_max_power
        sqrt_Pmax = np.sqrt(P_max_uav)
        P_max_cu = dbm_2_watt(a.cu_max_power_dbm)
        F_max = a.bs_max_freq
        D_max_u = a.uav_max_delay
        D_max_c = a.cu_max_delay
        D_sen = a.uav_sen_duration
        V_min_tau = a.uav_min_speed * a.time_slot_duration
        V_max_tau = a.uav_max_speed * a.time_slot_duration

        # ── 连续动作头拆分 ──
        # 波束 = 方向 (2N raw → L2 normalize) + 功率 (scalar ∈ [0,1])
        # C6/C8 自动满足：||w||² = power × P_max ≤ P_max
        # f_u/f_c 由 C13/C14 取等的闭式解给出，不占动作维度
        self.bs_continuous_action_splits = {
            "uav_angles": I,
            "uav_distances": I,
            "uav_off_durations": I,
            "uav_sen_beam_dir_real": I * N,
            "uav_sen_beam_dir_imag": I * N,
            "uav_sen_beam_power": I,
            "uav_off_beam_dir_real": I * N,
            "uav_off_beam_dir_imag": I * N,
            "uav_off_beam_power": I,
            "cu_off_powers": J,
            "cu_off_durations": J,
        }

        # ── 连续动作边界 ──
        bs_continuous_low = np.concatenate([
            np.zeros(I, dtype=np.float32),                              # uav_angles
            np.full(I, V_min_tau, dtype=np.float32),                   # uav_distances
            np.full(I, epsilon, dtype=np.float32),                     # uav_off_durations
            np.full(I * N, -1.0, dtype=np.float32),                    # uav_sen_beam_dir_real
            np.full(I * N, -1.0, dtype=np.float32),                    # uav_sen_beam_dir_imag
            np.zeros(I, dtype=np.float32),                             # uav_sen_beam_power  [0,1]
            np.full(I * N, -1.0, dtype=np.float32),                    # uav_off_beam_dir_real
            np.full(I * N, -1.0, dtype=np.float32),                    # uav_off_beam_dir_imag
            np.zeros(I, dtype=np.float32),                             # uav_off_beam_power  [0,1]
            np.full(J, epsilon, dtype=np.float32),                     # cu_off_powers
            np.full(J, epsilon, dtype=np.float32),                     # cu_off_durations
        ])
        bs_continuous_high = np.concatenate([
            np.full(I, 2.0 * np.pi, dtype=np.float32),                 # uav_angles
            np.full(I, V_max_tau, dtype=np.float32),                   # uav_distances
            np.full(I, D_max_u - D_sen, dtype=np.float32),             # uav_off_durations
            np.full(I * N, 1.0, dtype=np.float32),                     # uav_sen_beam_dir_real
            np.full(I * N, 1.0, dtype=np.float32),                     # uav_sen_beam_dir_imag
            np.ones(I, dtype=np.float32),                              # uav_sen_beam_power  [0,1]
            np.full(I * N, 1.0, dtype=np.float32),                     # uav_off_beam_dir_real
            np.full(I * N, 1.0, dtype=np.float32),                     # uav_off_beam_dir_imag
            np.ones(I, dtype=np.float32),                              # uav_off_beam_power  [0,1]
            np.full(J, P_max_cu, dtype=np.float32),                    # cu_off_powers
            np.full(J, D_max_c, dtype=np.float32),                     # cu_off_durations
        ])

        # ── 离散动作：I 个 UAV 各选 1 个 CU ──
        self.bs_discrete_action_dims = np.full(I, J, dtype=np.int64)

        # ── 覆盖动作空间 ──
        self.action_space = {
            "bs": {
                "continuous": spaces.Box(
                    low=bs_continuous_low, high=bs_continuous_high, dtype=np.float32
                ),
                "discrete": spaces.MultiDiscrete(self.bs_discrete_action_dims),
            }
        }

    @staticmethod
    def _build_beam_vectors(real_part, imag_part):
        """将实部/虚部拼接为复波束向量列表（每 UAV 一个 N 维复数向量）。"""
        return [
            real_part[i] + 1j * imag_part[i]
            for i in range(len(real_part))
        ]

    @staticmethod
    def _build_beam_from_dir_power(dir_real_flat, dir_imag_flat, power, I, N, P_max):
        """方向+功率分解 → C6/C8 自动满足的波束向量。

        :param dir_real_flat: (I*N,)  各 UAV 的波束方向实部（raw, 将被归一化）
        :param dir_imag_flat: (I*N,)  各 UAV 的波束方向虚部
        :param power:         (I,)    ∈ [0, 1] 功率比例因子
        :param I, N, P_max:   场景参数
        :return: list of (N,) complex arrays — 最终波束向量组
        """
        beams = []
        for i in range(I):
            d_real = dir_real_flat[i * N : (i + 1) * N]
            d_imag = dir_imag_flat[i * N : (i + 1) * N]
            d_vec = d_real + 1j * d_imag
            d_norm = float(np.linalg.norm(d_vec))
            if d_norm < 1e-12:
                d_vec = np.zeros(N, dtype=complex)
                d_vec[0] = 1.0 + 0j
                d_norm = 1.0
            d_unit = d_vec / d_norm                       # unit-norm direction
            pwr = float(np.clip(power[i], 0.0, 1.0))
            beams.append(d_unit * np.sqrt(pwr * P_max))   # 自动满足 ||w||² = pwr × P_max ≤ P_max
        return beams

    def _parse_actions(self, continuous_actions, discrete_actions):
        """将归一化动作为物理量，返回所有 P0 变量的 dict。"""
        a = self.base_args
        I = a.uavs_num
        J = a.cus_num
        N = a.antenna_nums

        # ── 连续动作缩放 ──
        cont_low = self.action_space["bs"]["continuous"].low
        cont_high = self.action_space["bs"]["continuous"].high
        cont = np.clip(np.asarray(continuous_actions, dtype=np.float32), 0.0, 1.0)
        cont = cont * (cont_high - cont_low) + cont_low

        offset = 0
        diff_theta = cont[offset : offset + I]; offset += I
        diff_distance = cont[offset : offset + I]; offset += I
        uav_off_duration = cont[offset : offset + I]; offset += I

        sen_beam_dir_real_flat = cont[offset : offset + I * N]; offset += I * N
        sen_beam_dir_imag_flat = cont[offset : offset + I * N]; offset += I * N
        sen_beam_power = cont[offset : offset + I]; offset += I
        off_beam_dir_real_flat = cont[offset : offset + I * N]; offset += I * N
        off_beam_dir_imag_flat = cont[offset : offset + I * N]; offset += I * N
        off_beam_power = cont[offset : offset + I]; offset += I

        cu_off_power = cont[offset : offset + J]; offset += J
        cu_off_duration = cont[offset : offset + J]; offset += J

        # ── 波束向量构造：方向归一化 + 功率缩放 → C6/C8 自动满足 ──
        P_max_uav = a.uav_max_power
        uav_sen_beam_vectors = self._build_beam_from_dir_power(
            sen_beam_dir_real_flat, sen_beam_dir_imag_flat,
            sen_beam_power, I, N, P_max_uav,
        )
        uav_off_beam_vectors = self._build_beam_from_dir_power(
            off_beam_dir_real_flat, off_beam_dir_imag_flat,
            off_beam_power, I, N, P_max_uav,
        )

        # ── 下一时刻位置 ──
        next_uavs_pos = self.cur_uavs_pos + np.stack(
            [
                diff_distance * np.cos(diff_theta),
                diff_distance * np.sin(diff_theta),
                np.zeros_like(diff_distance),
            ],
            axis=1,
        )

        # ═══ C15 由闭式解保证，C6/C8 由方向+功率分解保证，C13/C14 由闭式解保证 ═══

        # ── UAV-CU 匹配矩阵 ──
        discrete = np.clip(np.asarray(discrete_actions, dtype=np.int64), 0, J - 1)
        uavs_cus_matched = np.zeros((I, J), dtype=np.float32)
        for uav_idx, cu_idx in enumerate(discrete):
            uavs_cus_matched[uav_idx, int(cu_idx)] = 1.0

        return {
            "diff_theta": diff_theta,
            "diff_distance": diff_distance,
            "next_uavs_pos": next_uavs_pos,
            "uav_off_duration": uav_off_duration,
            "cu_off_power": cu_off_power,
            "cu_off_duration": cu_off_duration,
            "uav_sen_beam_vectors": uav_sen_beam_vectors,
            "uav_off_beam_vectors": uav_off_beam_vectors,
            # f_u/f_c 由 reward_compute 内部闭式解计算，非 agent 输出
            "uavs_cus_matched": uavs_cus_matched,
            "discrete": discrete,
        }

    def step(self, actions, i_episode=None):
        """
        执行一步 P0 端到端仿真。
        返回值格式与 MyEnv.step 一致：
            (next_state, total_reward, reward_dict, done, E_sum, success_flag)
        """
        bs_actions = actions["bs"] if isinstance(actions, dict) and "bs" in actions else actions
        continuous_raw = np.asarray(bs_actions["continuous"], dtype=np.float32)
        discrete_raw = np.asarray(bs_actions["discrete"], dtype=np.int64)

        # ── 解析动作 ──
        parsed = self._parse_actions(continuous_raw, discrete_raw)

        # ── UAV 位置更新 ──
        prev_uavs_pos = self.cur_uavs_pos.copy()

        # ── 调用 PureMyReward ──
        total_reward, reward, energy_opt = self.reward_calculator.reward_compute(
            uavs_2_cus_channels=self.uavs_2_cus_channels,
            uavs_2_bs_channels=self.uavs_2_bs_channels,
            cus_2_bs_channels=self.cus_2_bs_channels,
            uavs_2_targets_channels=self.uavs_2_targets_channels,
            uavs_targets_matched_matrix=self.uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=parsed["uavs_cus_matched"],
            uavs_pos=prev_uavs_pos,
            uavs_pos_cur=parsed["next_uavs_pos"],
            uavs_off_duration=parsed["uav_off_duration"],
            cus_off_power=parsed["cu_off_power"],
            cus_entertaining_task_size=self.cus_entertaining_task_size,
            uav_sen_beam_vectors=parsed["uav_sen_beam_vectors"],
            uav_off_beam_vectors=parsed["uav_off_beam_vectors"],
            # f_u/f_c 由 reward_compute 内部 C13/C14 取等闭式解计算
            cus_off_durations=parsed["cu_off_duration"],
            feasibility_enforce=self.feasibility_enforce,
        )

        # ── 成功标志：无任何约束违反 ──
        comps = reward.get("components", {})
        has_violation = (
            comps.get("boundary_penalty", 0) > 0
            or comps.get("collision_penalty", 0) > 0
            or comps.get("bs_alloc_spectrum_penalty", 0) > 0
            or comps.get("tanh_penalty_raw", 0) > 2.0
        )
        success = int(not has_violation)

        # ── 推进环境状态 ──
        self.t += 1
        self.cur_uavs_pos = parsed["next_uavs_pos"]
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()
        self._refresh_channels()
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )

        # ── 打印 ──
        comps = reward.get("components", {})
        f_u_vals = comps.get("uav_bs_freqs", np.zeros(self.base_args.uavs_num))
        f_c_vals = comps.get("cu_bs_freqs", np.zeros(self.base_args.cus_num))
        color_reset = "\033[0m"
        color_title = "\033[38;5;67m"
        color_metric = "\033[38;5;109m"
        color_energy = "\033[38;5;108m"
        color_penalty = "\033[38;5;137m"
        print(f"==================================================")
        print(f"{color_title} ------------ [Pure P0 Action Details] ------------ {color_reset}")
        for i in range(self.base_args.uavs_num):
            print(
                f"  {color_metric}UAV-{i}{color_reset}: "
                f"angle={parsed['diff_theta'][i]:.4f} rad, "
                f"dist={parsed['diff_distance'][i]:.4f} m, "
                f"D_off={parsed['uav_off_duration'][i]:.4f} s, "
                f"f_u={f_u_vals[i]:.2e} Hz, "
                f"{color_energy}→ CU-{parsed['discrete'][i]}{color_reset}"
            )
        for j in range(self.base_args.cus_num):
            print(
                f"  {color_penalty}CU-{j}{color_reset}: "
                f"p={parsed['cu_off_power'][j]:.4f} W, "
                f"D_off={parsed['cu_off_duration'][j]:.4f} s, "
                f"f_c={f_c_vals[j]:.2e} Hz"
            )
        print(
            f"{color_title}Episode {i_episode}, Time Slot {self.t}:{color_reset} "
            f"{color_metric}Total Reward = {total_reward:.4f}{color_reset}, "
            f"{color_energy}E_sum = {energy_opt:.4f}{color_reset}, "
            f"{color_penalty}Has Violation = {has_violation}{color_reset}"
        )
        print(f"{color_title} ------------ [Penalty Summary] ------------ {color_reset}")
        print(f"  tanh_norm={comps.get('tanh_penalty', 0):.4f}  "
              f"tanh_raw={comps.get('tanh_penalty_raw', 0):.3f}  "
              f"extra_tanh={comps.get('extra_tanh_penalty', 0):.4f}  "
              f"penalty_kept(boundary+collision+spectrum)={comps.get('boundary_penalty',0)+comps.get('collision_penalty',0)+comps.get('bs_alloc_spectrum_penalty',0):.1f}")
        print(f"{color_title} ------------ [Energy Breakdown] ------------ {color_reset}")
        print(f"  E_fly    = {comps.get('E_fly', 0.0):.4f}")
        print(f"  E_sen    = {comps.get('E_sen', 0.0):.4f}")
        print(f"  E_off    = {comps.get('E_off', 0.0):.4f}")
        print(f"  E_cu_off = {comps.get('E_cu_off', 0.0):.4f}")
        print(f"  E_cp     = {comps.get('E_cp', 0.0):.4f}")
        # ── Top-K 约束违反诊断 ──
        tanh_detail = comps.get("tanh_detail", {})
        if tanh_detail:
            sorted_items = sorted(tanh_detail.items(), key=lambda kv: kv[1], reverse=True)
            k = 8
            print(f"{color_penalty} ------------ [Top-{k} Constraint Violations] ------------ {color_reset}")
            for name, val in sorted_items[:k]:
                marker = ">>>" if val > 0.5 else "  "
                print(f"  {marker} {name}: {val:.4f}")
        print(f"==================================================")

        next_state_dict = {"bs": self._build_bs_observation()}
        done = int(self.t >= self.madrl_args.total_time_slots)
        return next_state_dict, float(total_reward), reward, done, float(energy_opt), success

    def _build_bs_observation(self):
        """覆写父类，追加 sen_sinr (dB) 到观测向量末尾。"""
        parent_obs = super()._build_bs_observation()
        sinr_feature = np.float32(self.base_args.sen_sinr)
        return np.append(parent_obs, sinr_feature).astype(np.float32)

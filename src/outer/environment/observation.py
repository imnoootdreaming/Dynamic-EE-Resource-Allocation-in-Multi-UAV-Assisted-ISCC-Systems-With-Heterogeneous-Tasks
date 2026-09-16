"""BS 观测构建：具名分段清单（单一定义源）+ 相对几何量编码。

从 `environment/my_env.py` 抽离的、与环境状态机无关的纯计算，遵循与
`channel_models` / `topology` / `mapping` 相同的「无 self 纯函数」约定，
使 `MyEnv` 只负责状态机与 delegate，观测逻辑本身可单测。

设计要点
--------
1. **单一定义源**：`BS_OBSERVATION_SEGMENTS` 同时驱动 `compute_bs_obs_dim`
   与 `build_bs_observation`。「维度累加顺序」与「向量拼接顺序」由构造保证一致，
   不再依赖 my_env 中两处手工同步（原 `obs_dim_bs` 累加式 vs `_build_bs_observation` 拼接）。
2. **相对几何量**：不再投喂任何绝对坐标（原 K·3 全目标坐标 + I·3 UAV 坐标 +
   两组 I·3 窗口目标坐标，共 156 维已删除），改为
   - 裸距离（UAV→CU、CU→BS）；
   - 极坐标三元组 ``(d/R, sinθ, cosθ)``（UAV→当前窗口目标 / 下一窗口目标 / BS）。
   第 6/7/8 段的 θ 为**世界坐标系下的水平方位角**：本轮动作仍是绝对飞行角
   （`uav_angles ∈ [0, 2π]` 直接作用于世界坐标轴），策略需要绝对方位才能输出
   可执行角度，故此处刻意不改成机体系相对角（机体系相对化属后续 Phase 2）。
3. **尺度归一化**：所有距离统一除以 `radius`（`base_args.radius` ← 内层
   `deploy_radius`），量级受控，避免 MLP 面对米级原始量纲。
4. **编码形式**：选极坐标三元组而非笛卡尔 ``(Δx, Δy)``。与动作位移合成
   ``next_pos = cur_pos + d·(cosθ, sinθ)`` **同构**——策略只需把观测里的
   ``(sinθ, cosθ)`` 直接回吐为动作角即可「直线飞向目标」，需要拟合的映射退化为
   近似直通；同时 ``sin/cos`` 规避了 ``arctan2`` 在 ±π 处的跳变（不连续是 MLP
   最难拟合的形态）。

语义边界
--------
- BS 固定部署在坐标原点（与 `channel_models.compute_com_channel_gain` 中的
  ``bs_pos`` 一致），故 UAV→BS 的水平方位即「回中心方向」。
- 第 6/7/8 段的 ``d`` 取**水平投影距离**（UAV 与目标/BS 在 z 上不同高，而飞行位移
  只发生在 x-y 平面），使其与动作的可达位移直接可比；
  第 4/5 段取**三维欧氏距离**，因为它们服务于「选哪个 CU 卸载」的路损判断。
- UAV 与目标水平重合时（`generate_pos` 令 UAV 初始位于被分配目标的水平投影处）
  方向退化，此时取 ``(sinθ, cosθ) = (0, 1)`` 并令 ``d = 0``，避免 0/0。
"""

from typing import Any, Callable, NamedTuple, Tuple

import numpy as np

from environment.mapping import flatten_complex

# BS 固定部署在部署区域中心（=`base_args.center`=[0,0]），高度与地面节点同为 0。
BS_POSITION = np.zeros(3, dtype=np.float64)

# 论文默认场景规模（I=4, J=10, K=40, N=10）下的参考总维度。
# 仅用于「改动分段布局后立刻发现回归」的护栏；换规模做消融时不做硬断言。
REFERENCE_SCENE_SHAPE = {
    "uavs_num": 4,
    "cus_num": 10,
    "targets_num": 40,
    "antenna_nums": 10,
}
REFERENCE_OBS_DIM = 987

_EPS = 1e-12


class ObsContext(NamedTuple):
    """构建 BS 观测所需的只读环境快照（由 `MyEnv` 组装，本模块只读不改）。"""

    uavs_2_bs_channels: np.ndarray          # (I, 1, N) complex：时隙 t UAV→BS 信道
    uavs_2_cus_channels: np.ndarray         # (I, J, N) complex：时隙 t UAV→CU 信道
    cus_2_bs_channels: np.ndarray           # (J, 1) complex：时隙 t CU→BS 信道
    uavs_pos: np.ndarray                    # (I, 3) 当前 UAV 位置
    cus_pos: np.ndarray                     # (J, 3) 当前 CU 位置
    targets_pos: np.ndarray                 # (K, 3) 目标位置（静态）
    cur_target_indices: np.ndarray          # (I,) 当前窗口各 UAV 的分配目标索引
    next_target_indices: np.ndarray         # (I,) 下一窗口各 UAV 的分配目标索引
    slots_until_switch: np.ndarray          # (1,) 窗口切换倒计时
    radius: float                           # 部署半径 R，用于距离归一化


class ObsSegment(NamedTuple):
    """观测分段：`dim` 声明维度，`build` 产出等长一维向量，两者由自检核对。"""

    name: str
    dim: Callable[[Any], int]
    build: Callable[[ObsContext], np.ndarray]


def relative_polar_encoding(delta: np.ndarray, radius: float) -> np.ndarray:
    """位移向量 → 极坐标三元组 ``(d_h/R, sinθ, cosθ)``，按行展平。

    :param delta: (M, 3) 「起点→终点」位移（终点减起点）
    :param radius: 归一化尺度（部署半径 R）
    :return: (M*3,) float32，行主序 ``[d0, sinθ0, cosθ0, d1, sinθ1, cosθ1, ...]``

    其中 ``d_h`` 为水平投影距离，``θ = arctan2(Δy, Δx)`` 为世界系水平方位角。
    """
    delta = np.asarray(delta, dtype=np.float64).reshape(-1, 3)
    delta_h = delta[:, :2]
    distance_h = np.linalg.norm(delta_h, axis=1)

    # 水平方向退化（Δxy → 0）时取单位方向 (cosθ, sinθ) = (1, 0)，避免 0/0 与 NaN。
    degenerate = distance_h < _EPS
    safe_distance = np.where(degenerate, 1.0, distance_h)
    sin_theta = np.where(degenerate, 0.0, delta_h[:, 1] / safe_distance)
    cos_theta = np.where(degenerate, 1.0, delta_h[:, 0] / safe_distance)

    scale = max(float(radius), _EPS)
    return np.stack(
        [distance_h / scale, sin_theta, cos_theta], axis=1
    ).astype(np.float32).reshape(-1)


def _normalized_distance(origins: np.ndarray, destinations: np.ndarray, radius: float) -> np.ndarray:
    """(M,3) × (N,3) → (M,N) 三维欧氏距离 / R。"""
    delta = np.asarray(origins, dtype=np.float64)[:, None, :] - np.asarray(destinations, dtype=np.float64)[None, :, :]
    scale = max(float(radius), _EPS)
    return (np.linalg.norm(delta, axis=2) / scale).astype(np.float32)


def _clipped_target_positions(targets_pos: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """按索引取目标位置（含越界保护，保持 `TargetsNum-1` 上界语义）。"""
    targets_pos = np.asarray(targets_pos)
    clipped = np.clip(np.asarray(target_indices, dtype=np.int64), 0, targets_pos.shape[0] - 1)
    return targets_pos[clipped]


# ── 各分段构建函数（纯计算，不依赖 self / 不修改入参） ──────────────────────────

def _build_h_uav_bs(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.uavs_2_bs_channels)


def _build_h_uav_cu(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.uavs_2_cus_channels)


def _build_h_cu_bs(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.cus_2_bs_channels)


def _build_d_uav_cu(ctx: ObsContext) -> np.ndarray:
    return _normalized_distance(ctx.uavs_pos, ctx.cus_pos, ctx.radius).reshape(-1)


def _build_d_cu_bs(ctx: ObsContext) -> np.ndarray:
    return _normalized_distance(
        ctx.cus_pos, BS_POSITION[None, :], ctx.radius
    ).reshape(-1)


def _build_rel_uav_cur_target(ctx: ObsContext) -> np.ndarray:
    return relative_polar_encoding(
        _clipped_target_positions(ctx.targets_pos, ctx.cur_target_indices) - ctx.uavs_pos,
        ctx.radius,
    )


def _build_rel_uav_next_target(ctx: ObsContext) -> np.ndarray:
    return relative_polar_encoding(
        _clipped_target_positions(ctx.targets_pos, ctx.next_target_indices) - ctx.uavs_pos,
        ctx.radius,
    )


def _build_rel_uav_bs(ctx: ObsContext) -> np.ndarray:
    return relative_polar_encoding(BS_POSITION[None, :] - ctx.uavs_pos, ctx.radius)


def _build_slots_until_switch(ctx: ObsContext) -> np.ndarray:
    return np.asarray(ctx.slots_until_switch, dtype=np.float32).reshape(-1)


# ── 唯一真相来源：顺序 == 维度累加顺序 == 拼接顺序 ──────────────────────────────
BS_OBSERVATION_SEGMENTS: Tuple[ObsSegment, ...] = (
    ObsSegment(
        "h_uav_bs",
        lambda a: a.uavs_num * a.antenna_nums * 2,
        _build_h_uav_bs,
    ),
    ObsSegment(
        "h_uav_cu",
        lambda a: a.uavs_num * a.cus_num * a.antenna_nums * 2,
        _build_h_uav_cu,
    ),
    ObsSegment(
        "h_cu_bs",
        lambda a: a.cus_num * 2,
        _build_h_cu_bs,
    ),
    ObsSegment(
        "d_uav_cu",
        lambda a: a.uavs_num * a.cus_num,
        _build_d_uav_cu,
    ),
    ObsSegment(
        "d_cu_bs",
        lambda a: a.cus_num,
        _build_d_cu_bs,
    ),
    ObsSegment(
        "rel_uav_cur_target",
        lambda a: a.uavs_num * 3,
        _build_rel_uav_cur_target,
    ),
    ObsSegment(
        "rel_uav_next_target",
        lambda a: a.uavs_num * 3,
        _build_rel_uav_next_target,
    ),
    ObsSegment(
        "rel_uav_bs",
        lambda a: a.uavs_num * 3,
        _build_rel_uav_bs,
    ),
    ObsSegment(
        "slots_until_switch",
        lambda a: 1,
        _build_slots_until_switch,
    ),
)


def compute_bs_obs_dim(base_args) -> int:
    """按分段清单累加 BS 观测维度（与 `build_bs_observation` 共用同一清单）。"""
    return int(sum(segment.dim(base_args) for segment in BS_OBSERVATION_SEGMENTS))


def build_bs_observation(ctx: ObsContext) -> np.ndarray:
    """按分段清单顺序拼接 BS 观测，返回 (dim,) float32（默认场景 987 维）。"""
    return np.concatenate(
        [segment.build(ctx) for segment in BS_OBSERVATION_SEGMENTS]
    ).astype(np.float32)


def segment_dims(base_args) -> Tuple[Tuple[str, int], ...]:
    """返回 ``((段名, 维度), ...)``，供日志打印与维度核对。"""
    return tuple((segment.name, int(segment.dim(base_args))) for segment in BS_OBSERVATION_SEGMENTS)


def format_bs_observation_layout(base_args) -> str:
    """渲染分段维度核对表（含逐段维度与合计）。"""
    dims = segment_dims(base_args)
    width = max(len(name) for name, _ in dims)
    lines = [
        "[BS Observation Layout] {} 段，合计 {} 维".format(
            len(dims), sum(dim for _, dim in dims)
        )
    ]
    for index, (name, dim) in enumerate(dims, start=1):
        lines.append("  {:>2}. {:<{width}} : {:>4}".format(index, name, dim, width=width))
    return "\n".join(lines)


def validate_bs_observation_layout(base_args, ctx: ObsContext) -> Tuple[Tuple[str, int, int], ...]:
    """逐段核对「声明维度」与「实际拼接长度」，返回 ``(段名, 声明维度, 实际长度)``。

    清单式单一数据源已消除「维度累加顺序」与「拼接顺序」错位的可能，但仍可能出现
    「单个 segment.build 产出长度与其 dim() 不一致」的手误，故用真实上下文各构建一次
    做端到端核对。
    """
    return tuple(
        (
            segment.name,
            int(segment.dim(base_args)),
            int(np.asarray(segment.build(ctx)).reshape(-1).shape[0]),
        )
        for segment in BS_OBSERVATION_SEGMENTS
    )


def is_reference_scene(base_args) -> bool:
    """判断当前场景规模是否等于论文默认配置（决定是否做 987 维硬断言）。"""
    return all(
        int(getattr(base_args, key)) == value for key, value in REFERENCE_SCENE_SHAPE.items()
    )

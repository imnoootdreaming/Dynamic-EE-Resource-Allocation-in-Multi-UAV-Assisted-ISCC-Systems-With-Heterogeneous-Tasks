"""BS observation layout: channels, target sequences, tasks, and motion geometry."""

from typing import Any, Callable, NamedTuple, Tuple

import numpy as np

from environment.mapping import flatten_complex

BS_OBSERVATION_CONTRACT = "bs_target_sequence_cu_velocity_uav_pairs_prev_displacement_xyz_v3"

# 论文默认场景规模（I=4, J=10, K=40, N=10）下的参考总维度。
# 仅用于「改动分段布局后立刻发现回归」的护栏；换规模做消融时不做硬断言。
REFERENCE_SCENE_SHAPE = {
    "uavs_num": 4,
    "cus_num": 10,
    "targets_num": 40,
    "antenna_nums": 10,
}
REFERENCE_OBS_DIM = 1151


class ObsContext(NamedTuple):
    """构建 BS 观测所需的只读环境快照（由 `MyEnv` 组装，本模块只读不改）。"""

    uavs_2_bs_channels: np.ndarray          # (I, 1, N) complex：时隙 t UAV→BS 信道
    uavs_2_cus_channels: np.ndarray         # (I, J, N) complex：时隙 t UAV→CU 信道
    cus_2_bs_channels: np.ndarray           # (J, 1) complex：时隙 t CU→BS 信道
    uavs_pos: np.ndarray                    # (I, 3) 当前 UAV 位置
    targets_pos: np.ndarray                 # (K, 3) static target positions
    target_sequence_indices: np.ndarray     # (I, W), ordered by visit
    current_target_window: int              # zero-based, terminal clamped to last
    slots_until_switch: np.ndarray          # (1,) 窗口切换倒计时
    task_sizes: np.ndarray                  # (J,) L_j(t)
    task_cycles: np.ndarray                 # (J,) C_j(t)
    task_delays: np.ndarray                 # (J,) D_j^max(t)
    cus_velocities: np.ndarray              # (J, 3) 当前 CU 速度
    uav_previous_displacement: np.ndarray  # (I, 3) 上一槽 UAV 位移，首槽为零


class ObsSegment(NamedTuple):
    """观测分段：`dim` 声明维度，`build` 产出等长一维向量，两者由自检核对。"""

    name: str
    dim: Callable[[Any], int]
    build: Callable[[ObsContext], np.ndarray]


def _build_h_uav_bs(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.uavs_2_bs_channels)


def _build_h_uav_cu(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.uavs_2_cus_channels)


def _build_h_cu_bs(ctx: ObsContext) -> np.ndarray:
    return flatten_complex(ctx.cus_2_bs_channels)


def _build_target_relative_sequence(ctx: ObsContext) -> np.ndarray:
    """Raw 3D target minus current UAV position, ordered UAV/target/xyz."""
    targets = np.asarray(ctx.targets_pos)[np.asarray(ctx.target_sequence_indices, dtype=np.int64)]
    return (targets - np.asarray(ctx.uavs_pos)[:, None, :]).astype(np.float32).reshape(-1)


def _build_target_current_one_hot(ctx: ObsContext) -> np.ndarray:
    indices = np.asarray(ctx.target_sequence_indices)
    one_hot = np.zeros(indices.shape, dtype=np.float32)
    one_hot[:, ctx.current_target_window] = 1.0
    return one_hot.reshape(-1)


def _build_slots_until_switch(ctx: ObsContext) -> np.ndarray:
    return np.asarray(ctx.slots_until_switch, dtype=np.float32).reshape(-1)


def _build_task_sizes(ctx: ObsContext) -> np.ndarray:
    return np.asarray(ctx.task_sizes, dtype=np.float32).reshape(-1)


def _build_task_cycles(ctx: ObsContext) -> np.ndarray:
    return np.asarray(ctx.task_cycles, dtype=np.float32).reshape(-1)


def _build_task_delays(ctx: ObsContext) -> np.ndarray:
    return np.asarray(ctx.task_delays, dtype=np.float32).reshape(-1)


def _build_cu_velocities(ctx: ObsContext) -> np.ndarray:
    """Current CU velocities, ordered by CU then x/y/z."""
    return np.asarray(ctx.cus_velocities, dtype=np.float32).reshape(-1)


def _build_uav_relative_positions(ctx: ObsContext) -> np.ndarray:
    """UAV j minus UAV i for each i < j, in lexicographic pair/xyz order."""
    positions = np.asarray(ctx.uavs_pos)
    i, j = np.triu_indices(len(positions), k=1)
    return (positions[j] - positions[i]).astype(np.float32).reshape(-1)


def _build_uav_previous_displacement(ctx: ObsContext) -> np.ndarray:
    """Previous slot's UAV displacement, ordered by UAV then x/y/z."""
    return np.asarray(ctx.uav_previous_displacement, dtype=np.float32).reshape(-1)


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
    ObsSegment("target_relative_sequence", lambda a: a.targets_num * 3,
               _build_target_relative_sequence),
    ObsSegment("target_current_one_hot", lambda a: a.targets_num,
               _build_target_current_one_hot),
    ObsSegment(
        "slots_until_switch",
        lambda a: 1,
        _build_slots_until_switch,
    ),
    ObsSegment("task_sizes", lambda a: a.cus_num, _build_task_sizes),
    ObsSegment("task_cycles", lambda a: a.cus_num, _build_task_cycles),
    ObsSegment("task_delays", lambda a: a.cus_num, _build_task_delays),
    ObsSegment("cu_velocities", lambda a: 3 * a.cus_num, _build_cu_velocities),
    ObsSegment("uav_relative_positions",
               lambda a: 3 * a.uavs_num * (a.uavs_num - 1) // 2,
               _build_uav_relative_positions),
    ObsSegment("uav_previous_displacement", lambda a: 3 * a.uavs_num,
               _build_uav_previous_displacement),
)


def compute_bs_obs_dim(base_args) -> int:
    """按分段清单累加 BS 观测维度（与 `build_bs_observation` 共用同一清单）。"""
    return int(sum(segment.dim(base_args) for segment in BS_OBSERVATION_SEGMENTS))


def build_bs_observation(ctx: ObsContext) -> np.ndarray:
    """按分段清单顺序拼接 BS 观测，返回 (dim,) float32（默认场景 1151 维）。"""
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
    """判断当前场景规模是否等于论文默认配置（决定是否做参考维度硬断言）。"""
    return all(
        int(getattr(base_args, key)) == value for key, value in REFERENCE_SCENE_SHAPE.items()
    )

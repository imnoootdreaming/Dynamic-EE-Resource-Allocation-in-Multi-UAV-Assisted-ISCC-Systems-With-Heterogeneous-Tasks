"""外层环境随机种子管理器（三层确定性派生）。

动机：原实现中 CU 马尔可夫轨迹与 NLoS 信道分量直接使用
``np.random.default_rng(base_args.seed)``，导致：

1. **跨采样进程相同**——``_set_seed`` 里的 ``np.random.seed()`` 只作用于旧式全局
   RandomState，管不到 ``default_rng`` 创建的独立 Generator，各并行 rollout 的
   场景完全同分布；
2. **跨 episode 相同**——所有随机量只在 ``MyEnv.__init__`` 一次性预生成，
   ``reset()`` 仅复位，同一进程内每个 episode 的场景动态完全重复。

本模块统一以显式种子分层派生，保证「跨 worker 互异、跨 episode 互异、
同参数可复现」三层性质：

::

    base_args.seed (全局基准)
      ├─ 学习进程 setSeed(global_seed)
      ├─ worker_env_seed = seed + (worker_id + 1)      # 复用现有 seed_offset 偏移
      └─ episode_rng    = SeedSequence(worker_env_seed, spawn_key=(episode_idx,))
                          -> np.random.Generator       # 布点 / CU 轨迹 / NLoS / 调度

worker_id 从 0 起，与 ``MHBPPO_main.py`` 启动采样进程时的编号一致；
``+1`` 与既有 ``seed_offset = worker_id + 1`` 口径保持一致（学习进程自用
``global_seed``，采样进程基线从 +1 开始偏移，互不重叠）。
"""

import numpy as np


class SeedManager:
    """三层种子派生：global / worker_env / episode。

    所有方法均为纯函数式确定性派生——同一 ``base_seed`` 下任何时刻、任何进程
    调用都会得到相同的种子/Generator，因此并行采样与逐 episode 重生成的场景
    均可严格复现。
    """

    def __init__(self, base_seed: int):
        self._base_seed = int(base_seed)

    @property
    def global_seed(self) -> int:
        """学习进程（Learner）使用的全局种子，等于 base_seed。"""
        return self._base_seed

    def worker_env_seed(self, worker_id: int) -> int:
        """第 worker_id 个采样进程的场景基线种子。

        ``worker_env_seed = base_seed + (worker_id + 1)``，与采样进程
        ``_set_seed(base_args.seed + seed_offset)`` 的既有口径一致。
        """
        return self._base_seed + int(worker_id) + 1

    def episode_seed_sequence(self, worker_id: int, episode_idx: int) -> np.random.SeedSequence:
        """逐 episode 派生的 SeedSequence。

        用 ``spawn_key=(episode_idx,)`` 而非 ``seed + episode_idx`` 线性偏移，
        由 SeedSequence 的熵搅混保证不同 (worker, episode) 组合的流互不碰撞。
        """
        return np.random.SeedSequence(entropy=self.worker_env_seed(worker_id),
                                      spawn_key=(int(episode_idx),))

    def episode_rng(self, worker_id: int, episode_idx: int) -> np.random.Generator:
        """返回第 worker_id 个采样进程第 episode_idx 个 episode 的独立 Generator。

        场景重生成（布点 / CU 轨迹 / NLoS）统一从该 Generator 取随机数。
        """
        return np.random.default_rng(self.episode_seed_sequence(worker_id, episode_idx))


def scenario_rng(base_seed: int, seed_offset: int, episode_idx: int) -> np.random.Generator:
    """便捷函数：不经过 SeedManager 实例，直接按 (seed_offset, episode_idx) 派生。

    :param base_seed: ``base_args.seed`` 全局基准。
    :param seed_offset: 采样进程的种子偏移（``worker_id + 1``；0 表示学习进程
        /维度探测用的主进程场景，即基线 seed 本身）。
    :param episode_idx: episode 序号（0 起）。
    """
    entropy = int(base_seed) + int(seed_offset)
    return np.random.default_rng(np.random.SeedSequence(entropy=entropy,
                                                        spawn_key=(int(episode_idx),)))

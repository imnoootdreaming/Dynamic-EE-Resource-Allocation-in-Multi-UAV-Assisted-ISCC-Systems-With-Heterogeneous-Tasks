"""外层基础参数适配层（base params）。

数据源：`src/inner/parameters.py` 中定义的 `Parameters`（论文 Simulation Setup 参数）。
本模块**不修改** inner 任何代码，只按文件路径以独立别名加载其 `Parameters` 类，并把内层
原生字段名适配成旧外层 env / reward 使用的字段名，同时补齐内层缺失的外层专用字段。

为什么需要适配层：
    旧 env（`environment/my_env.py`）与 reward（`reward/my_reward.py`）大量使用
    `uavs_num / cus_num / targets_num / radius / antenna_nums / frac_d_lambda /
    alpha_uav_link / alpha_cu_link / radar_rcs / sen_sinr / time_slot_duration /
    z_scale / markov_*` 等字段名，而内层 `Parameters` 用 `I / J / K / deploy_radius /
    N / d_over_lambda / alpha_1 / alpha_3 / xi_0 / eps_sinr_db / tau_slot` 等命名。
    `BaseArgsAdapter` 在两者之间做一层 O(1) 查表映射，使 env / reward 副本几乎零逻辑改动。

命名冲突规避：
    内层模块顶层名为 `parameters`，而 `reward/cccp_bridge.py` 会执行
    ``from parameters import Parameters``。因此本模块命名为 ``base_params.py``，并用
    importlib 以别名 ``_iscc_inner_parameters`` 按路径加载内层文件，避免污染
    ``sys.modules['parameters']``，也避免与外层同名模块互相顶替。
"""

import importlib.util
import os
import sys

# ── 路径定位：src/outer/configs/base_params.py -> src/inner/parameters.py ──────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/outer/configs
_OUTER_DIR = os.path.dirname(_THIS_DIR)                      # src/outer
_SRC_DIR = os.path.dirname(_OUTER_DIR)                       # src
_INNER_DIR = os.path.join(_SRC_DIR, "inner")                 # src/inner
_INNER_PARAMS_PATH = os.path.join(_INNER_DIR, "parameters.py")
_INNER_PARAMS_ALIAS = "_iscc_inner_parameters"


def _load_inner_parameters_class():
    """按文件路径以别名加载内层 `parameters.py` 的 `Parameters` 类。

    用独立模块名加载（而非 ``from parameters import Parameters``），避免与外层可能存在的
    同名模块、以及 cccp_bridge 后续导入的内层 ``parameters`` 相互干扰。
    """
    if _INNER_PARAMS_ALIAS in sys.modules:
        return sys.modules[_INNER_PARAMS_ALIAS].Parameters
    if not os.path.isfile(_INNER_PARAMS_PATH):
        raise ImportError("未找到内层参数文件：{}".format(_INNER_PARAMS_PATH))
    spec = importlib.util.spec_from_file_location(_INNER_PARAMS_ALIAS, _INNER_PARAMS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_INNER_PARAMS_ALIAS] = module
    spec.loader.exec_module(module)
    return module.Parameters


# ── 旧外层字段名 -> 内层 Parameters 原生字段名 ────────────────────────────────
# 仅覆盖 env / reward 实际使用到的字段；未列出的内层原生字段（I/J/K/N/tau_slot...）
# 由 BaseArgsAdapter.__getattr__ 透传，可直接访问。
_LEGACY_TO_INNER = {
    # 场景规模
    "uavs_num": "I",
    "cus_num": "J",
    "targets_num": "K",
    "radius": "deploy_radius",
    "uav_height": "uav_height",
    # UAV 飞行
    "uav_max_speed": "uav_max_speed",
    "uav_min_speed": "uav_min_speed",
    "uav_safe_distance": "uav_safe_distance",
    "uav_c1": "varrho_1",
    "uav_c2": "varrho_2",
    # 信道
    "bandwidth": "B",
    "noise_power_density_dbm": "noise_power_density_dbm",
    "ref_path_loss_db": "ref_path_loss_db",
    "rician_factor_db": "rician_factor_db",
    "alpha_uav_link": "alpha_1",   # 旧实现 UAV-CU 与 UAV-BS 同指数
    "alpha_cu_link": "alpha_3",
    "antenna_nums": "N",
    "frac_d_lambda": "d_over_lambda",
    # 感知
    "radar_rcs": "xi_0",
    "sen_sinr": "eps_sinr_db",
    "radar_duty_ratio": "delta_radar",
    "var_range_fluctuation": "sigma_pre_sq",
    "radar_impulse_duration": "nu_pulse",
    "radar_spectrum_shape": "gamma_radar",
    "uav_sen_duration": "D_bar_sen",
    # BS 计算
    "bs_cycles_per_bit": "C_cycles_per_bit",
    "bs_max_freq": "F_max",
    "kappa": "kappa_cpu",
    # 任务 / 时延 / 功率
    "cu_max_power_dbm": "P_max_cu_dbm",
    "uav_max_power": "P_max_uav",   # 线性 W（cccp_bridge 会 _watt_2_dbm 回读）
    "cu_max_delay": "D_max_cu_delay",
    "uav_max_delay": "D_max_sen",
    "time_slot_duration": "tau_slot",
    # 权重 / 随机种子
    "omega_weight_1": "omega_1",
    "omega_weight_2": "omega_2",
    "omega_weight_3": "omega_3",
    "seed": "seed",
}


def _make_extra_defaults():
    """内层 `Parameters` 缺失、但外层 env / reward 需要的字段默认值。

    取旧 `get_base_args()` 的默认值，保证 env（马尔可夫 CU 轨迹、部署中心）与
    reward / cccp_bridge（z_scale 归一化口径）行为不变。每次调用返回**全新**实例，
    避免调用方原地修改污染后续返回值。
    """
    return {
        "z_scale": 1e5,                                   # my_reward / cccp_bridge 归一化口径
        "center": [0.0, 0.0],                             # 部署区域中心（BS 位于原点）
        "num_cases": 30,                                  # 与旧 argparse 默认保持一致
        "markov_velocity": [1.0, 0.0, 0.0],               # CU 马尔可夫初始速度
        "markov_memory_level": 0.4,                       # CU 马尔可夫记忆水平
        "markov_asymptotic_mean_of_velocity": [1.0, 0.0, 0.0],
        "markov_standard_deviation_of_velocity": 2.0,
    }


class BaseArgsAdapter:
    """把内层 `Parameters` 适配成旧外层 env / reward 使用的字段名。

    读取（``__getattr__``）优先级：
        1) 旧字段名映射表 `_LEGACY_TO_INNER`；
        2) 补齐的扩展字段 `_extra`（z_scale / markov_* / center ...）；
        3) 内层 `Parameters` 原生字段（I/J/K/N/tau_slot ...）透传。

    写入（``__setattr__``）透传到底层 `Parameters`（字段存在时），否则写入扩展字典，
    以便需要时覆盖参数（例如测试脚本修改 ``uav_height`` / ``seed``）。
    """

    def __init__(self, params, extra=None):
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_extra", dict(extra) if extra else {})
        object.__setattr__(self, "_alias", dict(_LEGACY_TO_INNER))

    # -- 读 ---------------------------------------------------------------
    def __getattr__(self, name):
        # 仅当常规属性查找失败时触发（_params/_extra/_alias 已由 object.__setattr__ 落到实例字典）
        params = object.__getattribute__(self, "_params")
        alias = object.__getattribute__(self, "_alias")
        extra = object.__getattribute__(self, "_extra")

        if name in alias:
            return getattr(params, alias[name])
        if name in extra:
            # 返回存储对象本体（与原 argparse.Namespace 语义一致，允许原地读取/修改）
            return extra[name]
        if hasattr(params, name):
            return getattr(params, name)
        raise AttributeError(
            "BaseArgsAdapter 无法解析字段 '{}'（既不在映射表、扩展字段，也不在内层 Parameters 中）".format(name)
        )

    # -- 写 ---------------------------------------------------------------
    def __setattr__(self, name, value):
        params = object.__getattribute__(self, "_params")
        alias = object.__getattribute__(self, "_alias")
        extra = object.__getattribute__(self, "_extra")

        if name in alias and hasattr(params, alias[name]):
            setattr(params, alias[name], value)
            return
        if hasattr(params, name):
            setattr(params, name, value)
            return
        extra[name] = value

    # -- 调试 -------------------------------------------------------------
    def as_dict(self):
        """导出适配后的关键字段快照，便于日志 / 排查。"""
        keys = list(self._alias.keys()) + list(self._extra.keys())
        return {key: getattr(self, key) for key in keys}

    def __repr__(self):
        return "BaseArgsAdapter(params={!r})".format(self._extra.get("_params_name", "Parameters"))


def get_base_args(**overrides):
    """构建以 inner `Parameters` 为数据源的外层参数对象。

    :param overrides: 可选的关键字覆盖（键为旧字段名或内层字段名），用于测试 / 实验变体。
    :return: `BaseArgsAdapter` 实例，可直接替代原 `get_base_args()` 返回的 Namespace。
    """
    params_cls = _load_inner_parameters_class()
    params = params_cls()
    adapter = BaseArgsAdapter(params, _make_extra_defaults())
    for key, value in overrides.items():
        setattr(adapter, key, value)
    return adapter

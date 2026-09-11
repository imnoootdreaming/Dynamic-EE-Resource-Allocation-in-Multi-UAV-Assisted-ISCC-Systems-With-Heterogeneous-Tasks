"""约束包：自动发现并汇总本目录下的所有约束模块。

约定：本目录下每个模块（除 __init__.py 和以 `_` 开头的模块）需提供
    NAME  : str            —— 论文中的约束标签，如 "P2:Sensing-SINR"
    build(ctx) -> list     —— 返回该约束对应的 cvxpy 约束列表
新增约束只需在本目录新建一个符合上述约定的文件，无需修改任何注册代码。
"""

import importlib
import pkgutil

_MODULE_LIST = []
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name.startswith("_"):
        continue
    _module = importlib.import_module("{}.{}".format(__name__, _module_info.name))
    if hasattr(_module, "NAME") and hasattr(_module, "build"):
        _MODULE_LIST.append(_module)

_MODULE_LIST.sort(key=lambda module: module.__name__)


def constraint_names():
    """按文件名顺序返回所有已注册约束的论文标签。"""
    return [module.NAME for module in _MODULE_LIST]


def collect_constraints(ctx):
    """按文件名顺序汇总所有约束模块生成的 cvxpy 约束。"""
    constraints = []
    for module in _MODULE_LIST:
        constraints.extend(module.build(ctx))
    return constraints

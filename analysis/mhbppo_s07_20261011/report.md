# S07实施验收（2026-10-11）

结论：两项批准改动已实施，43项单测、80步真实求解/采样与两次PPO更新通过，语法与diff whitespace检查通过。

## 改动

- 接收波束mask按头名动态计算；effective alpha0排除两个头概率/熵，专有梯度None；alpha>0恢复，默认alpha1。保持参数布局，同规模旧state_dict严格加载；sample/evaluate/BC/KL同源，consumer拒绝概率契约不符的旧rollout。
- 新公共target_layout.py计算每扇区独立圆周半径；参数默认sector_ring/1/3/.85/clockwise；外层顺序调度、首目标上方初始化，内层独立入口一致。区域/UAV规模变化自动计算，非法参数或初始碰撞拒绝。
- 窗口长度统一到varkappa，target_hold_slots为读写别名；保留原随机布局调用兼容、保留旧target RNG消耗，使同seed通信随机样本可对照。
- 动作概率metadata记录有效heads/dims；scene.target_layout记录实际半径/圆心/预算/方向；source指纹包含共享几何文件。

## 验证

解释器ML2024_Task1 Python3.9.21，Torch2.5.1，NumPy1.25.2，MOSEK11.1.10。unittest discover共43tests，exit0；覆盖alpha0/.5/1、Beta/Gaussian、多I、恢复Adam动量冻结、旧参数加载、概率错配拒绝、600/200与4/8组合、I1/n1/n2、方向/半径/扇区/初始位置、窗口别名序列化、内外层一致、通信rng一致以及既有19项诊断测试。

真实spawn sampler每个alpha一个40步episode，validate_sample比对maskmetadata及数值logprob与learner evaluate一致；PPO更新所有参数有限、scalar有限，alpha0 beam专有权重不变且gradNone，alpha1专有权重改变。双JSONL各40条，逐键对齐。

随机初始化的运行检查：alpha0 optimal5/40，alpha1 optimal0/40。这些仅用于运行契约验证，不代表训练后表现，也没有证据承诺optimal上升或100%。默认alpha1保留网络波束优化能力，但仍需用户重新训练评估。

默认半径74.26776m、邻距45.9m；区域200/I4半径26.65261m；区域600/I8/K80半径49.72310m。更改区域应使用deploy_radius，增加I时需同步K或时域满足调度契约。

## 文件与作用域

summary.json、source_manifest.json、implementation.diff.txt、unit_tests.stderr.txt及integration_summary.json为验收证据。真实运行日志路径见integration_summary.json。before/保存本次开始时相关源码，包含用户已有改动；未回退或覆盖旧改动。既有历史布局训练和构造统计保留历史作用域，不外推新布局。

S07_USAGE.md为入口与参数说明。未启动长期训练或停止用户训练。
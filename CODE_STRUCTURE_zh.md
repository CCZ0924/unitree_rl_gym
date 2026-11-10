# Unitree RL Gym 代码结构与文件说明

> 以 G1 训练为例，逐文件说明这个仓库的代码组织方式。
> 项目路径（容器内）：`/catkin_ws/src/unitree_rl_gym`
> 项目路径（宿主机）：`/home/ccz/docker_catkin_ws/unitree_RL_tutorial/src/unitree_rl_gym`
> 创建时间：2026-11-08

本文档只讲代码结构；环境安装、训练/部署命令和日志操作见 [DOCKER_USAGE_zh.md](DOCKER_USAGE_zh.md)。

---

## 目录

- [1. 总览与调用链](#1-总览与调用链)
- [2. 入口与调度](#2-入口与调度)
- [3. 配置层](#3-配置层)
- [4. 环境实现层](#4-环境实现层)
- [5. 工具层](#5-工具层)
- [6. 资源与产物](#6-资源与产物)
- [7. 仓库之外但训练必需](#7-仓库之外但训练必需)
- [8. 一次 G1 训练的完整执行过程](#8-一次-g1-训练的完整执行过程)
- [9. 常见修改对照表](#9-常见修改对照表)
- [10. 关键概念与注意事项](#10-关键概念与注意事项)
- [附录：完整文件树](#附录完整文件树)

---

## 1. 总览与调用链

训练命令：

```bash
python legged_gym/scripts/train.py --task=g1 --headless
```

调用链：

```
train.py
 ├─ helpers.get_args()                    解析 --task=g1、--headless 等
 ├─ legged_gym/envs/__init__.py           注册 g1 → G1Robot + G1RoughCfg + G1RoughCfgPPO
 ├─ task_registry.make_env("g1")
 │   ├─ g1_config.py                      取出 G1 配置
 │   ├─ helpers.parse_sim_params()        转成 Isaac Gym 仿真参数
 │   └─ G1Robot(...)                      g1_env.py
 │       └─ LeggedRobot.__init__           legged_robot.py
 │           ├─ BaseTask.__init__          base_task.py：仿真、viewer、基础缓冲区
 │           ├─ create_sim()               _create_ground_plane + _create_envs（加载 URDF）
 │           ├─ _init_buffers()            状态张量、PD 增益、默认关节角
 │           └─ _prepare_reward_function() 绑定奖励函数
 ├─ task_registry.make_alg_runner()
 │   ├─ 创建 logs/g1/<时间>_<run_name>/
 │   └─ OnPolicyRunner(...)                rsl_rl：PPO 训练器
 └─ ppo_runner.learn()                     rsl_rl 训练循环 ↔ env.step()
```

分层理解：

| 层 | 文件 | 职责 |
|---|---|---|
| 入口层 | `train.py` / `play.py` | 解析参数、串联流程 |
| 调度层 | `task_registry.py` / `helpers.py` | 创建环境与训练器、加载 checkpoint |
| 配置层 | `*_config.py` | 所有可调参数 |
| 环境层 | `legged_robot.py` / `g1_env.py` | 仿真交互、观测、奖励、终止、重置 |
| 算法层 | `rsl_rl`（仓库外） | PPO 训练循环与网络 |

---

## 2. 入口与调度

### 2.1 [legged_gym/scripts/train.py](legged_gym/scripts/train.py)

**训练入口**，只有 18 行，做三件事：

```python
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```

它本身不含任何训练逻辑，真正的循环在 `rsl_rl` 的 `OnPolicyRunner` 里。

### 2.2 [legged_gym/scripts/play.py](legged_gym/scripts/play.py)

**回放入口**，行为与训练不同：

| 行为 | 说明 |
|---|---|
| 强制加载模型 | 设置 `train_cfg.runner.resume = True` |
| 测试模式 | `env_cfg.env.test = True`，物理时间实时同步（会 sleep） |
| 关闭随机化 | 关闭观测噪声、摩擦随机化、推力 |
| 限制环境数 | 默认把环境数压到最多 100 个 |
| 缩短地形 | `num_rows`/`num_cols` 设为 5（当前 G1 是平面，实际影响不大） |
| 跑 10 个回合 | `for i in range(10 * max_episode_length)` |
| 导出策略 | 调用 `export_policy_as_jit()` 输出部署用的 `.pt` |

导出位置：

```
logs/g1/exported/policies/policy_lstm_1.pt   # LSTM 策略（G1 默认）
logs/g1/exported/policies/policy_1.pt        # MLP 策略
```

### 2.3 [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py)

**任务注册表**。导入各机型的配置类和环境类，然后注册：

| task 名 | 环境类 | 配置文件 |
|---|---|---|
| `go2` | `LeggedRobot`（基类，没有专用 env） | `go2_config.py` |
| `h1` | `H1Robot` | `h1_config.py` |
| `h1_2` | `H1_2Robot` | `h1_2_config.py` |
| `g1` | `G1Robot` | `g1_config.py` |

`task_registry` 中每个 task 保存三元组：`(环境类, 环境配置实例, PPO 配置实例)`。

### 2.4 [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)

**工厂类**，负责把配置变成可运行的对象。

| 方法 | 作用 |
|---|---|
| `register()` | 注册 task |
| `get_cfgs()` | 取出环境配置和 PPO 配置 |
| `make_env()` | 应用命令行覆盖（`--num_envs` 等）→ `parse_sim_params()` → 实例化环境 |
| `make_alg_runner()` | 创建日志目录 `logs/<experiment>/<MonDD_HH-MM-SS>_<run_name>` → 构造 `OnPolicyRunner` → 若 `--resume` 则加载 checkpoint |

日志目录命名使用 `datetime.now().strftime('%b%d_%H-%M-%S')`，容器时区是 UTC。

### 2.5 [legged_gym/utils/helpers.py](legged_gym/utils/helpers.py)

**公共工具函数**：

| 函数/类 | 作用 |
|---|---|
| `class_to_dict()` / `update_class_from_dict()` | 嵌套配置类 ↔ 字典互转，供 rsl_rl 与日志使用 |
| `set_seed()` | 固定 Python / NumPy / PyTorch 随机种子 |
| `parse_sim_params()` | 把配置里的 `sim` 转成 `gymapi.SimParams` |
| `get_load_path()` | 根据 `load_run` / `checkpoint` 解析出 checkpoint 路径 |
| `update_cfg_from_args()` | 用命令行参数覆盖配置 |
| `get_args()` | 定义所有命令行参数（`--task`、`--num_envs`、`--resume` 等） |
| `export_policy_as_jit()` | 导出部署策略：MLP 用 `torch.jit.script(actor)` |
| `PolicyExporterLSTM` | LSTM 策略导出器，会保存并更新 `hidden_state`/`cell_state` |

---

## 3. 配置层

### 3.1 [legged_gym/envs/base/base_config.py](legged_gym/envs/base/base_config.py)

很小的基类：递归地把嵌套配置类实例化成对象，使得配置可以写成：

```python
cfg.env.num_envs
cfg.rewards.scales.tracking_lin_vel
```

而不是操作字典。

### 3.2 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)

**所有机型的配置模板**，包含两个顶层配置类：

- `LeggedRobotCfg`：环境与机器人参数
- `LeggedRobotCfgPPO`：算法与训练参数

`LeggedRobotCfg` 的分块：

| 配置块 | 关键内容 |
|---|---|
| `env` | `num_envs=4096`、`num_observations=48`、`num_actions=12`、`episode_length_s=20`、`env_spacing=3` |
| `terrain` | `mesh_type='plane'`，地面摩擦/弹性；地形生成模块当前未启用 |
| `commands` | 指令范围 x/y ±1 m/s、yaw ±1 rad/s、heading ±π；`resampling_time=10` 秒；`heading_command=True` |
| `init_state` | 初始位置/姿态/速度、默认关节角 |
| `control` | 控制类型 `P`、刚度/阻尼、`action_scale`、`decimation` |
| `asset` | URDF 路径、脚部名称、碰撞惩罚部位、终止碰撞部位、自碰撞 |
| `domain_rand` | 摩擦、质量、推力等域随机化 |
| `rewards` | 奖励权重 `scales`、`tracking_sigma`、`only_positive_rewards`、各类软限位 |
| `normalization` | 观测缩放、clip 阈值 |
| `noise` | 观测噪声开关与各分量噪声尺度 |
| `viewer` | 相机位置 |
| `sim` | `dt=0.005`、substeps、PhysX 求解器参数 |

`LeggedRobotCfgPPO` 的分块：

| 配置块 | 关键内容 |
|---|---|
| `policy` | 默认 `ActorCritic`、隐藏层维度、激活函数 |
| `algorithm` | PPO 超参（学习率、clip、epoch 等） |
| `runner` | `num_steps_per_env=24`、`max_iterations=1500`、`save_interval=50`、实验名、resume 设置 |

### 3.3 [legged_gym/envs/g1/g1_config.py](legged_gym/envs/g1/g1_config.py)

**G1 的最终配置**，继承模板并按需覆盖。当前内容：

| 配置块 | G1 的值 |
|---|---|
| `init_state` | 初始高度 `z=0.8`；12 个默认关节角（hip_pitch -0.1、knee 0.3、ankle_pitch -0.2 等） |
| `env` | 观测 47、特权观测 50、动作 12 |
| `domain_rand` | 摩擦 `[0.1, 1.25]`、附加质量 `[-1, 3]`、每 5 秒随机推力 |
| `control` | stiffness：hip 100 / knee 150 / ankle 40；damping：hip 2 / knee 4 / ankle 2；`action_scale=0.25`；`decimation=4` |
| `asset` | `g1_12dof.urdf`、脚部名称 `ankle_roll`、碰撞惩罚 hip/knee、pelvis 接触终止、开启自碰撞 |
| `rewards` | G1 专用奖励权重，`base_height_target=0.78` |
| `G1RoughCfgPPO` | `ActorCriticRecurrent`、LSTM 64、MLP 32、熵系数 0.01、`max_iterations=10000`、`experiment_name='g1'` |

> **注意**：类名叫 `G1RoughCfg`（沿用了上游 legged_gym 的命名），但 G1 没有覆盖 `terrain`，
> 基类默认 `mesh_type='plane'`，因此**当前实际是在平地上训练**。

G1 当前启用的奖励权重：

```
正向：tracking_lin_vel 1.0, tracking_ang_vel 0.5, alive 0.15, contact 0.18
惩罚：lin_vel_z -2, ang_vel_xy -0.05, orientation -1, base_height -10,
      dof_acc -2.5e-7, dof_vel -1e-3, action_rate -0.01,
      dof_pos_limits -5, hip_pos -1, contact_no_vel -0.2, feet_swing_height -20
关闭：feet_air_time 0, collision 0, termination -0.0
```

### 3.4 其他机型配置

| 文件 | 说明 |
|---|---|
| [legged_gym/envs/go2/go2_config.py](legged_gym/envs/go2/go2_config.py) | Go2 四足，使用基类 `LeggedRobot` |
| [legged_gym/envs/h1/h1_config.py](legged_gym/envs/h1/h1_config.py) | H1 人形，10 个动作 |
| [legged_gym/envs/h1_2/h1_2_config.py](legged_gym/envs/h1_2/h1_2_config.py) | H1_2 人形，12 个动作 |

---

## 4. 环境实现层

### 4.1 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)

Isaac Gym 的底层封装：

- 获取 `gymapi` 句柄
- 选择计算设备（GPU pipeline → CUDA，否则 CPU）
- 分配基础缓冲区：`obs_buf`、`rew_buf`、`reset_buf`、`episode_length_buf`、`time_out_buf`、`privileged_obs_buf`
- 调用 `create_sim()` 创建仿真
- 非 headless 模式下创建 viewer，注册 ESC 退出、V 暂停同步快捷键
- `render()`：处理窗口事件、刷新画面

### 4.2 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)

**整个训练环境的核心（727 行）**，所有机型共用。关键方法：

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | [22](legged_gym/envs/base/legged_robot.py#L22) | 解析配置 → 建仿真/环境 → 建缓冲区 → 绑定奖励 |
| `step` | [49](legged_gym/envs/base/legged_robot.py#L49) | 一次策略步：clip 动作 → 渲染 → 按 `decimation` 循环 PD 控制与物理仿真 → `post_physics_step` |
| `post_physics_step` | [82](legged_gym/envs/base/legged_robot.py#L82) | 刷新张量、更新基座速度、计算终止/观测/奖励、重置 |
| `check_termination` | [118](legged_gym/envs/base/legged_robot.py#L118) | 终止条件：指定部位接触（pelvis）、roll/pitch 超限、超时 |
| `reset_idx` | [126](legged_gym/envs/base/legged_robot.py#L126) | 重置位置/关节/指令，并把回合奖励写入 `extras`，供 TensorBoard 记录 |
| `compute_reward` | [163](legged_gym/envs/base/legged_robot.py#L163) | 各奖励项加权求和、负值截断、叠加 termination |
| `compute_observations` | [182](legged_gym/envs/base/legged_robot.py#L182) | 默认观测拼接（G1 会覆盖） |
| `create_sim` | [198](legged_gym/envs/base/legged_robot.py#L198) | 创建仿真、地面、所有环境 |
| `_process_rigid_shape_props` | [214](legged_gym/envs/base/legged_robot.py#L214) | 摩擦/弹性域随机化 |
| `_process_dof_props` | [239](legged_gym/envs/base/legged_robot.py#L239) | 写入 PD 增益、力矩/速度限制、armature |
| `_process_rigid_body_props` | [267](legged_gym/envs/base/legged_robot.py#L267) | 基座质量随机化 |
| `_post_physics_step_callback` | [280](legged_gym/envs/base/legged_robot.py#L280) | 定时重采样指令、处理 heading 指令、推机器人 |
| `_resample_commands` | [292](legged_gym/envs/base/legged_robot.py#L292) | 随机生成新的速度指令，过小的指令置零 |
| `_compute_torques` | [308](legged_gym/envs/base/legged_robot.py#L308) | 动作 → PD 目标 → 力矩，并 clip 到力矩上限 |
| `_reset_dofs` | [332](legged_gym/envs/base/legged_robot.py#L332) | 随机初始关节角（默认角的 0.5~1.5 倍） |
| `_reset_root_states` | [347](legged_gym/envs/base/legged_robot.py#L347) | 随机初始位姿 |
| `_push_robots` | [369](legged_gym/envs/base/legged_robot.py#L369) | 随机水平推力（域随机化） |
| `_get_noise_scale_vec` | [398](legged_gym/envs/base/legged_robot.py#L398) | 观测噪声向量（G1 覆盖） |
| `_init_buffers` | [423](legged_gym/envs/base/legged_robot.py#L423) | 获取 GPU 状态张量，建立关节/基座/接触力视图，计算默认关节角和 PD 增益 |
| `_prepare_reward_function` | [485](legged_gym/envs/base/legged_robot.py#L485) | 按配置绑定 `_reward_<name>`，系数乘 `dt` |
| `_create_ground_plane` | [510](legged_gym/envs/base/legged_robot.py#L510) | 添加地面 |
| `_create_envs` | [520](legged_gym/envs/base/legged_robot.py#L520) | 加载 URDF、创建 N 个 actor、设置 DOF/刚体属性、查找脚和终止碰撞体 |
| `_get_env_origins` | [607](legged_gym/envs/base/legged_robot.py#L607) | 按网格给机器人分配初始位置 |
| `_parse_cfg` | [623](legged_gym/envs/base/legged_robot.py#L623) | 计算控制周期 `dt`、观测缩放、奖励权重、指令范围、回合长度 |
| `_reward_*` | [637-727](legged_gym/envs/base/legged_robot.py#L637) | 通用奖励函数 |

通用奖励函数清单：

| 函数 | 含义 |
|---|---|
| `_reward_tracking_lin_vel` | 线速度指令跟踪，`exp(-error²/tracking_sigma)` |
| `_reward_tracking_ang_vel` | 角速度指令跟踪 |
| `_reward_lin_vel_z` | 惩罚竖直速度 |
| `_reward_ang_vel_xy` | 惩罚横滚/俯仰角速度 |
| `_reward_orientation` | 惩罚机身倾斜（重力投影 xy） |
| `_reward_base_height` | 惩罚机身高度偏离目标 |
| `_reward_torques` | 惩罚力矩 |
| `_reward_dof_vel` | 惩罚关节速度 |
| `_reward_dof_acc` | 惩罚关节加速度 |
| `_reward_action_rate` | 惩罚动作变化率 |
| `_reward_collision` | 惩罚指定部位碰撞 |
| `_reward_termination` | 终止惩罚/奖励 |
| `_reward_dof_pos_limits` | 惩罚接近关节位置限位 |
| `_reward_dof_vel_limits` | 惩罚接近关节速度限位 |
| `_reward_torque_limits` | 惩罚接近力矩限位 |
| `_reward_feet_air_time` | 奖励抬脚时间（步幅） |
| `_reward_stumble` | 惩罚脚踢到垂直面 |
| `_reward_stand_still` | 零指令时惩罚运动 |
| `_reward_feet_contact_forces` | 惩罚足底冲击力过大 |

### 4.3 [legged_gym/envs/g1/g1_env.py](legged_gym/envs/g1/g1_env.py)

**G1 的差异化实现（123 行）**，`class G1Robot(LeggedRobot)`：

| 方法 | 作用 |
|---|---|
| `_get_noise_scale_vec` | 按 G1 的 47 维观测结构配置噪声 |
| `_init_foot` | 获取 rigid body 状态，提取双脚位置和速度 |
| `_init_buffers` | 调用基类后追加脚的初始化 |
| `update_feet_state` | 每步刷新脚的位置/速度 |
| `_post_physics_step_callback` | 计算步态相位：周期 0.8 秒，左右腿相差 0.5 周期 |
| `compute_observations` | 拼接 G1 的 47 维观测和 50 维特权观测 |
| `_reward_contact` | 足底接触与步态相位是否一致 |
| `_reward_feet_swing_height` | 摆动脚高度接近 0.08 m |
| `_reward_alive` | 每步存活奖励，固定返回 1 |
| `_reward_contact_no_vel` | 脚着地时仍有速度的惩罚 |
| `_reward_hip_pos` | 髋关节 yaw/roll 角度平方惩罚（关节索引 `[1,2,7,8]`） |

**G1 观测布局（47 维）**：

| 区间 | 内容 | 维度 |
|---|---|---|
| `[0:3]` | 基座角速度 × `obs_scales.ang_vel` | 3 |
| `[3:6]` | 投影重力（机身姿态） | 3 |
| `[6:9]` | 速度指令 × `commands_scale` | 3 |
| `[9:21]` | `(dof_pos - default_dof_pos) × obs_scales.dof_pos` | 12 |
| `[21:33]` | `dof_vel × obs_scales.dof_vel` | 12 |
| `[33:45]` | 上一步动作 | 12 |
| `[45:47]` | sin(phase)、cos(phase) | 2 |

特权观测在前面额外加入基座线速度 3 维，共 50 维。

### 4.4 其他机型的环境类

| 文件 | 说明 |
|---|---|
| [legged_gym/envs/h1/h1_env.py](legged_gym/envs/h1/h1_env.py) | H1，结构与 `g1_env.py` 基本一致 |
| [legged_gym/envs/h1_2/h1_2_env.py](legged_gym/envs/h1_2/h1_2_env.py) | H1_2，同上 |

Go2 没有专用环境类，直接使用 `LeggedRobot`。

---

## 5. 工具层

| 文件 | 作用 |
|---|---|
| [legged_gym/__init__.py](legged_gym/__init__.py) | 定义 `LEGGED_GYM_ROOT_DIR`，所有资源路径和日志路径都基于它 |
| [legged_gym/utils/__init__.py](legged_gym/utils/__init__.py) | 统一导出工具函数和类 |
| [legged_gym/utils/math.py](legged_gym/utils/math.py) | 四元数 yaw 旋转、角度归一化、sqrt 分布随机采样 |
| [legged_gym/utils/isaacgym_utils.py](legged_gym/utils/isaacgym_utils.py) | 四元数转欧拉角，被 `legged_robot.py` 用来计算 roll/pitch |
| [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py) | 上游 legged_gym 的地形生成器（坡道/台阶/坑），**当前训练路径未使用**：`legged_robot.py` 只创建平面地面 |
| [legged_gym/utils/logger.py](legged_gym/utils/logger.py) | 老的简易日志类，**当前训练不使用**；训练曲线由 rsl_rl 写入 TensorBoard |

---

## 6. 资源与产物

| 路径 | 作用 |
|---|---|
| [resources/robots/g1_description/g1_12dof.urdf](resources/robots/g1_description/g1_12dof.urdf) | **G1 训练实际加载的机器人模型**（12 自由度腿） |
| `resources/robots/g1_description/meshes/` | URDF 引用的 STL/DAE 网格 |
| `resources/robots/g1_description/g1_12dof.xml`、`scene.xml` | MuJoCo 部署用模型，训练不用 |
| 同目录的 23/29 dof URDF/XML | 其他 G1 版本，当前训练配置不用 |
| [logs/](logs/) | `events.out.tfevents.*`（TensorBoard 曲线）、`model_<iter>.pt`（checkpoint）、`exported/policies/*.pt`（部署策略） |
| [deploy/](deploy/) | 部署相关：`pre_train`（官方策略）、`deploy_mujoco`（Sim2Sim）、`deploy_real`（Sim2Real） |
| [doc/](doc/) | 安装配置文档 |

---

## 7. 仓库之外但训练必需

这些代码不在 `unitree_rl_gym` 目录里，但训练时会用到（容器内位于 `/catkin_ws/src`）：

| 路径 | 作用 |
|---|---|
| `../rsl_rl/rsl_rl/runners/on_policy_runner.py` | **真正的训练循环**：采样、PPO 更新、TensorBoard、保存/加载 |
| `../rsl_rl/rsl_rl/algorithms/ppo.py` | PPO 算法实现 |
| `../rsl_rl/rsl_rl/modules/actor_critic_recurrent.py` | G1 使用的 LSTM Actor-Critic 网络 |
| `../rsl_rl/rsl_rl/storage/` | rollout 数据存储与 GAE |
| `../IsaacGym_Preview_4_Package/` | NVIDIA 物理仿真引擎 |

所以要理解“训练是怎么更新网络的”，需要看 `rsl_rl`；`unitree_rl_gym` 只负责环境和配置。

---

## 8. 一次 G1 训练的完整执行过程

1. `train.py` 解析 `--task=g1 --headless`
2. 导入 `legged_gym.envs`，G1 被注册为 `G1Robot + G1RoughCfg + G1RoughCfgPPO`
3. `make_env()` 取配置、应用命令行覆盖、生成 Isaac Gym 仿真参数
4. `G1Robot` 构造：
   - `LeggedRobot.__init__` → `_parse_cfg` 计算 `dt`、缩放和回合长度
   - `BaseTask.__init__` 创建仿真和基础缓冲区
   - `create_sim()` → `_create_ground_plane()` 添加地面
   - `_create_envs()` 加载 `g1_12dof.urdf`，创建 4096 个机器人，设置 PD 增益、摩擦、质量，识别脚（`ankle_roll`）和终止碰撞体（`pelvis`）
   - `_init_buffers()` 建立 GPU 状态张量、默认关节角、PD 增益
   - `_prepare_reward_function()` 绑定所有非零奖励项
5. `make_alg_runner()` 新建 `logs/g1/Nov08_xx-xx-xx_/`，创建 LSTM + PPO 训练器
6. `learn()` 开始循环，每轮采样 `24 × 4096` 个策略步
7. 每个策略步：
   - 动作 clip → `_compute_torques()` 计算 PD 力矩
   - 物理仿真 `decimation=4` 次（每次 0.005 s，共 0.02 s，即 50 Hz）
   - `post_physics_step()`：刷新状态 → 更新步态相位 → 重采样指令/推机器人 → 判断终止 → 重置 → 计算奖励和观测
8. 每 `save_interval=50` 次迭代保存 `model_<iter>.pt`
9. 训练正常结束时额外保存最终模型
10. 之后用 `play.py` 加载 checkpoint 查看效果，并导出 `policy_lstm_1.pt` 供 MuJoCo/实机使用

---

## 9. 常见修改对照表

| 我想改… | 应该看 |
|---|---|
| 机器人 URDF、脚部名称、终止部位 | [g1_config.py](legged_gym/envs/g1/g1_config.py) 的 `asset` |
| PD 增益、动作缩放、控制频率 | `g1_config.py` 的 `control` |
| 默认关节角、初始高度 | `g1_config.py` 的 `init_state` |
| 奖励权重 | `g1_config.py` 的 `rewards.scales` |
| 奖励计算逻辑 | [g1_env.py](legged_gym/envs/g1/g1_env.py)（G1 专用）或 [legged_robot.py](legged_gym/envs/base/legged_robot.py)（通用） |
| 观测内容、维度 | `g1_env.py` 的 `compute_observations` + `g1_config.py` 的 `env.num_observations` |
| 动作维度/控制方式 | `g1_config.py` 的 `env.num_actions` + `control` + `_compute_torques` |
| 速度指令范围 | [legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py) 的 `commands.ranges` |
| 终止条件 | [legged_robot.py:118](legged_gym/envs/base/legged_robot.py#L118) |
| 域随机化 | `g1_config.py` 的 `domain_rand` + `_process_*_props` / `_push_robots` |
| 网络结构、LSTM、训练轮数 | `g1_config.py` 的 `G1RoughCfgPPO` |
| 地面/地形 | `legged_robot_config.py` 的 `terrain`（当前是平面） |
| 命令行参数 | [helpers.py:122](legged_gym/utils/helpers.py#L122) |
| 训练循环、PPO、保存逻辑 | `../rsl_rl/rsl_rl/runners/on_policy_runner.py` |

**平时基本不用动的文件**：`base_task.py`、`math.py`、`isaacgym_utils.py`、`terrain.py`、`logger.py`。

日常调参 90% 的时间只涉及：

- [g1_config.py](legged_gym/envs/g1/g1_config.py)
- [g1_env.py](legged_gym/envs/g1/g1_env.py)
- 偶尔 [legged_robot.py](legged_gym/envs/base/legged_robot.py)

---

## 10. 关键概念与注意事项

### 10.1 控制频率与回合长度

| 参数 | 值 | 来源 |
|---|---|---|
| 物理步长 `sim.dt` | 0.005 s | `legged_robot_config.py` |
| 控制降频 `decimation` | 4 | `g1_config.py` |
| 策略控制周期 `dt` | 0.02 s（50 Hz） | `dt = sim.dt × decimation` |
| 回合时长 | 20 s | `env.episode_length_s` |
| 单回合策略步数 | 1000 | `20 / 0.02` |

### 10.2 奖励的执行方式

- `_prepare_reward_function()` 只保留**非零**的奖励项，并把系数乘以 `dt`
- 函数名必须与配置键名严格对应：配置 `tracking_lin_vel` ↔ 方法 `_reward_tracking_lin_vel`
- `only_positive_rewards=True` 时，总奖励的负值被截断为 0，termination 在截断后单独加
- 每项奖励累计到 `episode_sums`，回合结束时按回合时长归一化，写入 TensorBoard 的 `Episode/rew_*`

### 10.3 关于 checkpoint

- `model_<iter>.pt` 包含：网络权重、优化器状态、当前迭代数、附加信息
- 只有 `--resume` 才会加载 checkpoint；默认训练从头开始
- `deploy/pre_train/*/motion.pt` 是官方预训练策略，只有 Actor，不能续训
- `logs/g1/exported/policies/*.pt` 是 `play.py` 导出的推理模型，供 MuJoCo/实机使用

### 10.4 容易误解的几点

1. **`G1RoughCfg` 名字里有 Rough，但当前训练是平地**（基类 `terrain.mesh_type='plane'`）
2. **`terrain.py` 和 `logger.py` 是继承来的、当前未使用的代码**
3. **Go2 没有自己的 env 文件**，直接注册基类 `LeggedRobot`
4. **训练算法不在本仓库**，在 `rsl_rl` 里
5. **`play.py` 不会新建训练 run 目录**：它只加载 checkpoint 并导出策略，
   产物是 `logs/<task>/exported/policies/*.pt`；只有 `train.py` 会写 `model_*.pt` 和 TensorBoard 事件
6. **日志目录名使用容器时间（UTC）**，比北京时间慢 8 小时

---

## 附录：完整文件树

```
unitree_rl_gym/
├── setup.py                        # 包安装信息与依赖
├── README.md / README_zh.md        # 项目说明
├── DOCKER_USAGE_zh.md              # Docker 使用说明
├── CODE_STRUCTURE_zh.md            # 本文档
├── LICENSE
├── doc/
│   ├── setup_zh.md                 # 安装文档
│   └── setup_en.md
├── legged_gym/
│   ├── __init__.py                 # LEGGED_GYM_ROOT_DIR / LEGGED_GYM_ENVS_DIR
│   ├── scripts/
│   │   ├── train.py                # 训练入口
│   │   └── play.py                 # 回放 / 导出入口
│   ├── envs/
│   │   ├── __init__.py             # 任务注册
│   │   ├── base/
│   │   │   ├── base_config.py      # 递归实例化配置
│   │   │   ├── base_task.py        # Isaac Gym 基础任务类
│   │   │   ├── legged_robot_config.py  # 环境 + PPO 配置模板
│   │   │   └── legged_robot.py     # 核心环境实现（727 行）
│   │   ├── g1/
│   │   │   ├── g1_config.py        # G1 配置
│   │   │   └── g1_env.py           # G1 环境子类
│   │   ├── go2/
│   │   │   └── go2_config.py       # Go2 配置（使用基类环境）
│   │   ├── h1/
│   │   │   ├── h1_config.py
│   │   │   └── h1_env.py
│   │   └── h1_2/
│   │       ├── h1_2_config.py
│   │       └── h1_2_env.py
│   └── utils/
│       ├── __init__.py             # 统一导出
│       ├── helpers.py              # 参数、配置转换、checkpoint、策略导出
│       ├── task_registry.py        # 任务注册与环境/训练器工厂
│       ├── terrain.py              # 地形生成（当前未使用）
│       ├── logger.py               # 旧日志类（当前未使用）
│       ├── math.py                 # 数学辅助
│       └── isaacgym_utils.py       # Isaac Gym 辅助
├── resources/
│   └── robots/
│       ├── g1_description/         # G1 URDF/XML + meshes
│       ├── go2/                    # Go2 模型
│       ├── h1/                     # H1 模型
│       └── h1_2/                   # H1_2 模型
├── deploy/
│   ├── pre_train/                  # 官方预训练策略 motion.pt
│   ├── deploy_mujoco/              # MuJoCo Sim2Sim
│   └── deploy_real/                # 实机部署（Python + C++）
└── logs/                           # 训练日志、模型、导出策略
```

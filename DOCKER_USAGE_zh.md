# Unitree RL Gym 使用说明（Docker 版）

> 适用容器：`g1_tutorial_env`
> 项目路径（容器内）：`/catkin_ws/src/unitree_rl_gym`
> 项目路径（宿主机）：`/home/ccz/docker_catkin_ws/unitree_RL_tutorial/src/unitree_rl_gym`
> 文档创建时间：2026-11-08

本文档记录在本机 Docker 环境中训练、查看、部署 Unitree 机器人策略的完整操作方法。项目本身的介绍见 [README_zh.md](README_zh.md)。

---

## 目录

- [0. 一分钟速查](#0-一分钟速查)
- [1. 项目与容器信息](#1-项目与容器信息)
- [2. 进入容器与激活环境](#2-进入容器与激活环境)
- [3. 训练](#3-训练)
- [4. 查看训练情况](#4-查看训练情况)
- [5. Sim2Sim（MuJoCo）](#5-sim2simmujoco)
- [6. Sim2Real（实物部署）](#6-sim2real实物部署)
- [7. 日志与 checkpoint](#7-日志与-checkpoint)
- [8. 常用 Docker / 系统命令](#8-常用-docker--系统命令)
- [9. 故障排查](#9-故障排查)
- [10. 推荐工作流](#10-推荐工作流)
- [附录：G1 关键配置速查](#附录g1-关键配置速查)

---

## 0. 一分钟速查

| 我想做什么 | 命令（已在容器内并激活环境的前提下） |
|---|---|
| 正式训练 G1 | `python legged_gym/scripts/train.py --task=g1 --headless` |
| 看训练后的机器人 | `python legged_gym/scripts/play.py --task=g1 --num_envs=16` |
| 看指定 checkpoint | `python legged_gym/scripts/play.py --task=g1 --load_run=<run目录名> --checkpoint=<编号>` |
| 断点续训 | `python legged_gym/scripts/train.py --task=g1 --headless --resume --load_run=<run目录名> --checkpoint=<编号>` |
| 边训练边看画面 | `python legged_gym/scripts/train.py --task=g1 --num_envs=512` |
| 看训练曲线 | 启动 TensorBoard，浏览器打开 `http://<容器IP>:6006` |
| MuJoCo 验证 | `python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml` |
| 实机部署 | `python deploy/deploy_real/deploy_real.py <网卡名> g1.yaml` |

---

## 1. 项目与容器信息

### 1.1 项目是什么

`unitree_rl_gym` 是宇树官方的强化学习运动控制示例仓库，基于 Isaac Gym + legged_gym + rsl_rl（PPO），支持：

- **Go2**（四足）
- **G1**（人形，12 自由度腿部）
- **H1**
- **H1_2**

完整流程：

```
Train（Isaac Gym 训练）
  → Play（回放、导出策略）
  → Sim2Sim（MuJoCo 验证）
  → Sim2Real（真机部署）
```

### 1.2 当前容器

| 项目 | 值 |
|---|---|
| 容器名 | `g1_tutorial_env` |
| 镜像 | `nvidia/cuda:12.8.0-devel-ubuntu22.04` |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8 GB） |
| 容器内用户 | root |
| Conda | `/miniconda`，使用环境 `unitree-rl`（Python 3.8） |
| 显示 | `DISPLAY=:0`，已挂载 `/tmp/.X11-unix`，`QT_X11_NO_MITSHM=1` |
| 网络 | bridge，IP 通常为 `172.17.0.2`（重启后可能变化，用命令查询） |

挂载关系：

| 宿主机 | 容器内 |
|---|---|
| `/home/ccz/docker_catkin_ws/unitree_RL_tutorial/src` | `/catkin_ws/src` |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` |

`/catkin_ws/src` 下与本项目相关的还有：

- `unitree_rl_gym` — 本项目
- `IsaacGym_Preview_4_Package` — Isaac Gym
- `rsl_rl` — 强化学习算法库

> **注意**
> 1. 本项目放在 `src` 下，但**不是 ROS/catkin 包**，不需要 `catkin_make`。
> 2. Conda 环境 `/miniconda/envs/unitree-rl` 位于容器可写层，**删除容器会丢失整个环境**。代码和日志在挂载目录里，不会丢。不要轻易 `docker rm` 这个容器；如需备份，可参考 [8.5 给容器做快照](#85-给容器做快照)。

---

## 2. 进入容器与激活环境

### 2.1 进入容器

```bash
docker exec -it g1_tutorial_env bash
```

进入后**每次都要**激活 Python 环境：

```bash
source /miniconda/etc/profile.d/conda.sh
conda activate unitree-rl
cd /catkin_ws/src/unitree_rl_gym
```

> 容器里直接敲 `python` 默认是 base 环境的 Python 3.14，**不能**用来跑本项目。
> 训练/回放必须使用 `unitree-rl` 环境（Python 3.8 + PyTorch 2.3.1）。

### 2.2 一条命令进入并激活

```bash
docker exec -it -w /catkin_ws/src/unitree_rl_gym g1_tutorial_env \
  bash -c 'source /miniconda/etc/profile.d/conda.sh && conda activate unitree-rl && exec bash'
```

### 2.3 验证环境

```bash
python -c "import isaacgym; import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

预期输出包含 `torch 2.3.1+cu121 cuda True`。

> 注意导入顺序：**必须先 `import isaacgym` 再 `import torch`**，否则会报
> `PyTorch was imported before isaacgym modules`。

### 2.4 不进入容器、直接执行单条命令

```bash
docker exec -w /catkin_ws/src/unitree_rl_gym g1_tutorial_env \
  bash -c 'source /miniconda/etc/profile.d/conda.sh && conda activate unitree-rl && \
  python legged_gym/scripts/play.py --task=g1 --num_envs=16'
```

---

## 3. 训练

所有命令默认在项目根目录 `/catkin_ws/src/unitree_rl_gym` 下执行，且已激活 `unitree-rl`。

### 3.1 正式训练（推荐，无画面）

```bash
python legged_gym/scripts/train.py --task=g1 --headless
```

- 默认并行 **4096** 个环境，训练 **10000** 次迭代
- 在本机 RTX 4060 上约 **4~6 小时**（实际取决于散热/功耗）
- 训练过程中不会显示 3D 画面，速度最快

### 3.2 后台训练（关闭终端也不中断）

```bash
docker exec -d -w /catkin_ws/src/unitree_rl_gym g1_tutorial_env bash -c '
source /miniconda/etc/profile.d/conda.sh && conda activate unitree-rl && \
python -u legged_gym/scripts/train.py --task=g1 --headless \
  > /catkin_ws/src/unitree_rl_gym/logs/train_g1.log 2>&1'
```

看日志（宿主机上直接看，因为目录是挂载的）：

```bash
tail -f /home/ccz/docker_catkin_ws/unitree_RL_tutorial/src/unitree_rl_gym/logs/train_g1.log
```

停止训练：

```bash
docker exec g1_tutorial_env pkill -f train.py
```

> 训练每 `save_interval`（默认 50）次迭代保存一次 checkpoint；正常结束时会额外保存最终模型。
> 用 `pkill`/`Ctrl+C` 强制中断时，最多可能丢掉最近 50 次迭代的进度。

### 3.3 边训练边看画面

```bash
# 去掉 --headless 即打开 Isaac Gym 窗口
python legged_gym/scripts/train.py --task=g1 --num_envs=512
```

- 不指定 `--num_envs` 时会用默认的 4096，窗口会非常卡，建议降到 256~1024
- 有画面时训练速度会明显下降，只适合调试，不适合正式训练

### 3.4 断点续训

> **默认行为（很重要）**：不加 `--resume` 时，每次训练都是**从头随机初始化**。
> 既不会自动接着上一次训练，也**不会**加载 `deploy/pre_train/` 下的预训练模型。
> `deploy/pre_train/*/motion.pt` 只用于 MuJoCo/实机部署，不参与训练初始化。

续训命令：

```bash
python legged_gym/scripts/train.py --task=g1 --headless --resume \
  --load_run=Nov08_10-31-52_ --checkpoint=0
```

- `--load_run` 填 `logs/g1/` 下的 run 目录名
- `--checkpoint` 填模型编号（不带 `.pt`）
- 续训会**新建一个 run 目录**保存后续结果，不会覆盖原来的 run
- 网络权重、优化器状态和迭代计数都会从 checkpoint 恢复
- 省略 `--load_run`/`--checkpoint` 时，会加载按**目录名字典序**排最后的 run 的最后一个模型；
  跨月份时这个顺序可能不对，建议总是显式指定

### 3.5 常用参数

| 参数 | 说明 | 示例 |
|---|---|---|
| `--task` | 必填，`go2` / `g1` / `h1` / `h1_2` | `--task=g1` |
| `--headless` | 关闭 3D 渲染，训练更快 | `--headless` |
| `--resume` | 从 checkpoint 继续训练 | `--resume` |
| `--experiment_name` | 实验名，决定 `logs/<name>` 目录 | `--experiment_name=g1` |
| `--run_name` | run 名，会拼在时间戳后面 | `--run_name=exp1` |
| `--load_run` | 续训/回放时指定 run 目录 | `--load_run=Nov08_10-31-52_` |
| `--checkpoint` | 指定 checkpoint 编号 | `--checkpoint=500` |
| `--num_envs` | 并行环境数，显存不够时调小 | `--num_envs=2048` |
| `--seed` | 随机种子 | `--seed=42` |
| `--max_iterations` | 最大迭代次数 | `--max_iterations=2000` |
| `--sim_device` | 仿真设备，可设 `cpu` | `--sim_device=cpu` |
| `--rl_device` | 训练设备，可设 `cpu` | `--rl_device=cpu` |

### 3.6 快速冒烟测试

刚改完配置、不确定环境是否正常时，先跑 10 次迭代验证：

```bash
python legged_gym/scripts/train.py --task=g1 --headless \
  --num_envs=512 --max_iterations=10
```

然后在 `logs/g1/` 下应能看到新目录、`model_0.pt` 和 TensorBoard 事件文件。

---

## 4. 查看训练情况

### 4.1 用 3D 画面回放策略（play.py）

```bash
# 默认加载 logs/g1 下最新 run 的最新 checkpoint
python legged_gym/scripts/play.py --task=g1 --num_envs=16

# 指定 run 和 checkpoint
python legged_gym/scripts/play.py --task=g1 --num_envs=16 \
  --load_run=Nov08_10-31-52_ --checkpoint=0
```

说明：

- 会打开 Isaac Gym 窗口，在宿主机桌面上显示
- 不指定 `--num_envs` 时最多显示 100 个机器人；建议 `--num_envs=1~16` 便于观察
- 回放会实时同步物理时间（`env_cfg.env.test = True`），G1 一局 20 秒、50 Hz，总共跑 10 局，约 **200 秒**后自动结束
- 关闭窗口或 `Ctrl+C` 可提前退出
- 每次运行都会导出推理策略到：
  - LSTM 策略：`logs/g1/exported/policies/policy_lstm_1.pt`（G1 默认是 LSTM）
  - MLP 策略：`logs/g1/exported/policies/policy_1.pt`

> 如果 checkpoint 训练次数很少（比如只有 `model_0.pt`），机器人会直接摔倒或抽搐，这是策略尚未学会走路的正常现象，不是环境坏了。

### 4.2 TensorBoard 看曲线

先启动 TensorBoard（后台运行）：

```bash
docker exec -d g1_tutorial_env bash -c '
source /miniconda/etc/profile.d/conda.sh && conda activate unitree-rl && \
tensorboard --logdir /catkin_ws/src/unitree_rl_gym/logs --bind_all --port 6006'
```

查询容器 IP：

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' g1_tutorial_env
```

假设输出 `172.17.0.2`，在**宿主机**浏览器打开：

```
http://172.17.0.2:6006
```

如果从另一台电脑访问，可在能访问宿主机的机器上做 SSH 隧道：

```bash
ssh -L 6006:172.17.0.2:6006 <用户名>@<宿主机IP>
# 然后本地浏览器打开 http://127.0.0.1:6006
```

停止 TensorBoard：

```bash
docker exec g1_tutorial_env pkill -f tensorboard
```

> **依赖修复**：本环境的 TensorBoard 2.14.0 最初是用 `--no-deps` 安装的，缺少
> `werkzeug`、`markdown`、`requests`、`tensorboard-data-server`、`google-auth`、
> `google-auth-oauthlib`、`grpcio`。如果重建环境后报 `No module named ...`，执行：
>
> ```bash
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
>   'werkzeug==2.2.3' 'markdown>=2.6.8' 'requests>=2.21,<3' \
>   'tensorboard-data-server>=0.7,<0.8' \
>   'google-auth<3,>=1.6.3' 'google-auth-oauthlib<1.1,>=0.5' 'grpcio>=1.48.2'
> ```
>
> 当前环境（2026-11-08）已补装并验证：`tensorboard --version` 正常，
> 宿主机访问 `http://172.17.0.2:6006` 返回 200。

重点关注的曲线：

| 曲线 | 含义 | 正常趋势 |
|---|---|---|
| `Train/mean_reward` | 平均回合奖励 | 持续上升 |
| `Train/mean_episode_length` | 平均存活步数 | 上升，最终接近 1000 |
| `Train/mean_reward/time` | 按墙钟时间看奖励 | 上升 |
| `Episode/rew_tracking_lin_vel` | 线速度跟踪奖励 | 上升 |
| `Episode/rew_tracking_ang_vel` | 角速度跟踪奖励 | 上升 |
| `Perf/total_fps` | 训练吞吐 | 本机约 7 万步/秒 |
| `Loss/value_function` | 价值函数损失 | 不应持续爆炸 |

> 训练初期奖励绝对值很小是正常的（奖励按物理步长缩放，且回合很快终止），看趋势即可。

### 4.3 命令行直接看最新指标

```bash
docker exec -w /catkin_ws/src/unitree_rl_gym g1_tutorial_env \
  /miniconda/envs/unitree-rl/bin/python -c '
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
for d in sorted(glob.glob("logs/g1/*")):
    ea = EventAccumulator(d); ea.Reload()
    tags = ea.Tags()["scalars"]
    if "Train/mean_reward" not in tags:
        continue
    r = ea.Scalars("Train/mean_reward")
    l = ea.Scalars("Train/mean_episode_length")
    print(f"{d}: iters={len(r)} reward={r[-1].value:.4f} ep_len={l[-1].value:.1f}")
'
```

### 4.4 查看某次 run 的详细信息

```bash
ls -lh logs/g1/<run目录名>/
cat logs/g1/<run目录名>/events.out.tfevents.* >/dev/null   # 仅确认文件存在
```

### 4.5 判断训练是否真的在跑

```bash
# 容器内进程
docker exec g1_tutorial_env pgrep -af 'train.py|play.py|tensorboard'

# GPU 占用
nvidia-smi

# 容器资源
docker stats g1_tutorial_env
```

如果 `pgrep` 没有输出，说明训练已经退出。

---

## 5. Sim2Sim（MuJoCo）

用 MuJoCo 快速验证策略效果，不依赖 Isaac Gym：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

- 配置文件：`deploy/deploy_mujoco/configs/g1.yaml`（还有 `h1.yaml`、`h1_2.yaml`）
- 默认加载官方预训练策略 `deploy/pre_train/g1/motion.pt`
- 会打开 MuJoCo 窗口，默认模拟 60 秒（`simulation_duration`）
- 需要 `DISPLAY`，当前容器已配置好

换成自己训练的模型：编辑 `deploy/deploy_mujoco/configs/g1.yaml`，修改 `policy_path`：

```yaml
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/g1/exported/policies/policy_lstm_1.pt"
```

> 注意：自己导出的是 LSTM 策略时文件名是 `policy_lstm_1.pt`，MLP 是 `policy_1.pt`。
> 同时确认 `kps`、`kds`、`default_angles`、`num_obs`、`num_actions`、`action_scale`
> 与训练时一致，否则策略效果会完全不对。

### 5.1 `deploy/pre_train/` 里是什么

`deploy/pre_train/{g1,h1,h1_2}/motion.pt` 是**宇树官方预先训练好的行走策略**，
采用 TorchScript（JIT）格式，只包含 Actor 网络（LSTM 64 + MLP 32），
**没有 Critic、优化器和迭代信息**，因此不能用于续训，只能做推理。

| 文件 | 观测维度 | 输出动作维度 |
|---|---|---|
| `pre_train/g1/motion.pt` | 47 | 12 |
| `pre_train/h1/motion.pt` | 41 | 10 |
| `pre_train/h1_2/motion.pt` | 47 | 12 |

用途：

- 作为 `deploy/deploy_mujoco/configs/*.yaml` 和 `deploy/deploy_real/configs/*.yaml`
  中 `policy_path` 的**默认值**，让 Sim2Sim 和实机部署无需自己训练即可运行
- `train.py` **不会**读取这些文件；训练默认从零开始
- 它们是状态化 LSTM，调用时会更新内部 `hidden_state`/`cell_state`，
  必须按控制周期连续调用；输入需要带 batch 维，形如 `[1, obs_dim]`

换成自己的策略时，只需把 yaml 里的 `policy_path` 指向
`logs/<task>/exported/policies/policy_lstm_1.pt`。

---

## 6. Sim2Real（实物部署）

> ⚠️ **安全第一**：真机首次测试务必吊装/扶稳机器人，准备好急停，先低速、小幅测试。
> 以下内容仅适用于你已经具备机器人调试权限和网络配置的情况。

### 6.1 前置条件

1. **安装 unitree_sdk2py**（当前 `unitree-rl` 环境里**没有安装**）：

   ```bash
   cd /catkin_ws/src
   git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
   cd unitree_sdk2_python
   pip install -e .
   ```

   如果目录已存在，直接：

   ```bash
   pip install -e /catkin_ws/src/unitree_sdk2_python
   ```

2. 机器人上电，处于**零力矩模式**
3. 按遥控器 `L2 + R2` 进入**调试模式**（关节处于阻尼状态）
4. 用网线连接电脑与机器人网口，配置电脑网卡 IP 到机器人同一网段
5. 用 `ifconfig` 或 `ip link` 查询与机器人相连的网卡名（如 `enp3s0`）

### 6.2 启动

```bash
python deploy/deploy_real/deploy_real.py enp3s0 g1.yaml
```

- 第一个参数：网卡名
- 第二个参数：`deploy/deploy_real/configs/` 下的配置文件名（`g1.yaml` / `h1.yaml` / `h1_2.yaml`）
- 启动后程序会先进入零力矩，再移动到默认姿态；按遥控器 `A` 开始执行策略，`select` 退出

### 6.3 换策略

编辑 `deploy/deploy_real/configs/g1.yaml`：

```yaml
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/g1/exported/policies/policy_lstm_1.pt"
```

注意真机配置里的 `kps`、`kds`、`default_angles`、`num_obs`、`num_actions`
必须与训练环境一致；G1 还包含手臂/腰部关节的保持参数。

### 6.4 C++ 版本（G1，需额外准备依赖）

`deploy/deploy_real/cpp_g1/` 下提供了 C++ 实现，适合对实时性要求更高的场景。但当前环境**没有**它需要的依赖：

- `libtorch`（CMake 期望位于 `deploy/deploy_real/cpp_g1/libtorch/share/cmake`）
- `unitree_sdk2` C++ 库、`ddscxx`、`iceoryx`
- `yaml-cpp`

准备齐以上依赖后，编译方式：

```bash
cd deploy/deploy_real/cpp_g1
mkdir -p build && cd build
cmake .. && make -j
```

日常调试建议优先使用 Python 版 `deploy_real.py`，依赖更少。

---

## 7. 日志与 checkpoint

### 7.1 目录结构

```
logs/
└── g1/                                    # experiment_name
    ├── Nov08_10-31-52_/                   # <月日_时分秒>_<run_name>
    │   ├── events.out.tfevents.*          # TensorBoard 事件
    │   ├── model_0.pt                     # 第 0 次迭代
    │   ├── model_50.pt                    # 每 50 次迭代保存一次
    │   └── ...
    └── exported/
        └── policies/
            ├── policy_lstm_1.pt           # LSTM 策略（G1）
            └── policy_1.pt                # MLP 策略
```

### 7.2 默认参数

| 参数 | 值 |
|---|---|
| `experiment_name` | `g1`（在 `legged_gym/envs/g1/g1_config.py` 中） |
| `max_iterations` | 10000 |
| `num_envs` | 4096 |
| `num_steps_per_env` | 24 |
| `save_interval` | 50（基类默认） |

### 7.3 加载规则

`play.py` / `--resume` 默认加载：

1. `logs/<experiment_name>/` 下按**目录名字典序**排序后的最后一个 run
2. 该 run 下编号最大的 `model_*.pt`

> 目录排序是字符串排序，不是按时间排序（源码中有 TODO）。跨月份时（例如 `Oct...` 与 `Nov...`）
> 可能选错 run，**建议显式指定 `--load_run` 和 `--checkpoint`**。

### 7.4 磁盘占用与清理

先看占用：

```bash
du -sh logs/ logs/g1/*/
df -h .
```

**删除单个 run：**

```bash
rm -rf logs/g1/<run目录名>
```

**只保留最新一次 run（其余全删）：**

```bash
# 1) 先预览会被删除的目录（exported 会被排除）
ls -dt logs/g1/*/ | grep -v '/exported/$'

# 2) 确认无误后执行：删除除最新 run 外的所有 run
ls -dt logs/g1/*/ | grep -v '/exported/$' | tail -n +2 | xargs -r rm -rf
```

**清空所有训练 run，但保留导出的策略：**

```bash
find logs/g1 -mindepth 1 -maxdepth 1 -type d ! -name exported -print    # 先预览
find logs/g1 -mindepth 1 -maxdepth 1 -type d ! -name exported -exec rm -rf {} +
```

**只删模型、保留 TensorBoard 曲线：**

```bash
find logs/g1 -name 'model_*.pt' -print     # 先预览
find logs/g1 -name 'model_*.pt' -delete
```

> 注意事项：
> 1. **训练进行中不要删除当前正在写的 run**，先 `pkill -f train.py`。
> 2. 删掉 `logs/g1/exported/` 后，MuJoCo/实机配置里指向 `policy_lstm_1.pt` 的路径会失效，
>    重新跑一次 `play.py` 即可再次导出。
> 3. 删除 run 后，`play.py` / `--resume` 默认加载的"最新 run"会随之改变。
> 4. 只保留 `events` 而删光 `model_*.pt` 后，无法回放和续训，只能看曲线。
> 5. 是否删除只看 run 的迭代数：只跑了几次迭代的短 run 基本没有使用价值，可以放心删。

### 7.5 备份模型

因为 `logs/` 在挂载目录里，宿主机可直接访问：

```bash
cp -r /home/ccz/docker_catkin_ws/unitree_RL_tutorial/src/unitree_rl_gym/logs/g1/exported \
      ~/g1_policy_backup/
```

---

## 8. 常用 Docker / 系统命令

### 8.1 容器生命周期

```bash
docker ps -a --filter name=g1_tutorial_env
docker start g1_tutorial_env
docker stop g1_tutorial_env          # 会终止容器内所有训练
docker restart g1_tutorial_env
docker logs --tail 100 g1_tutorial_env
docker stats g1_tutorial_env
```

> `docker stop` / `restart` 会直接杀掉正在运行的训练进程，非正常退出可能丢失最近 50 次迭代。

### 8.2 查看容器配置

```bash
docker inspect g1_tutorial_env --format '{{json .HostConfig.Binds}}'
docker inspect g1_tutorial_env --format '{{json .Config.Env}}'
docker inspect g1_tutorial_env --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
```

### 8.3 进程管理

```bash
# 查看
docker exec g1_tutorial_env pgrep -af 'train.py|play.py|tensorboard'

# 停止
docker exec g1_tutorial_env pkill -f train.py
docker exec g1_tutorial_env pkill -f play.py
docker exec g1_tutorial_env pkill -f tensorboard
```

### 8.4 显示环境检查

当前容器已满足 GUI 运行条件（X11 socket 已挂载、`DISPLAY=:0`、GPU 有 graphics 能力）。
可用以下命令自测：

```bash
docker exec g1_tutorial_env /miniconda/envs/unitree-rl/bin/python \
  -c "import tkinter; tkinter.Tk().destroy(); print('X OK')"
```

若输出 `X OK` 说明窗口可以正常弹出。

如果以后重建容器后无法显示，在宿主机执行：

```bash
xhost +local:root
```

并在 `docker run` 时加上：

```bash
-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw
```

### 8.5 给容器做快照

修改环境、安装新包之前，可以给容器做一次快照（镜像会占用磁盘）：

```bash
docker commit g1_tutorial_env g1_tutorial_env_backup_$(date +%Y%m%d)
```

恢复时用该镜像重新创建容器即可。

### 8.6 GPU 监控

```bash
nvidia-smi
watch -n 2 nvidia-smi
docker exec g1_tutorial_env nvidia-smi
```

---

## 9. 故障排查

| 现象 | 原因 | 解决方法 |
|---|---|---|
| `bash: python: command not found` | 没有激活 conda 环境 | `source /miniconda/etc/profile.d/conda.sh && conda activate unitree-rl` |
| `PyTorch was imported before isaacgym modules` | 导入顺序错误 | 先 `import isaacgym`，再 `import torch` |
| `ModuleNotFoundError: legged_gym` | 不在项目目录或未安装 | `cd /catkin_ws/src/unitree_rl_gym`，必要时 `pip install -e .` |
| `ValueError: No runs in this directory` | 没有对应实验的 checkpoint | 先训练，或检查 `--experiment_name` / `--load_run` |
| `CUDA out of memory` | 显存不足 | 降低 `--num_envs`（如 2048、1024） |
| 窗口不弹出 / `Could not initialize GLFW` | X11 或显示变量问题 | 用 [8.4](#84-显示环境检查) 自测；宿主机 `xhost +local:root` |
| TensorBoard 报 `No module named werkzeug/markdown` | 依赖缺失 | 按 [4.2](#42-tensorboard-看曲线) 的依赖修复命令安装 |
| TensorBoard 页面打不开 | 容器没有端口映射 | 用容器 IP `http://<IP>:6006`，或 SSH 隧道 |
| `play.py` 加载了旧的模型 | 默认按目录名字典序选 run | 显式指定 `--load_run` 和 `--checkpoint` |
| 机器人一回放就摔倒 | checkpoint 训练不足 | 继续训练；查看 `mean_reward` / `episode_length` 是否上升 |
| 训练速度很慢 | 开了 GUI 或环境数太多 | 加 `--headless`，降低 `--num_envs` |
| `No module named unitree_sdk2py` | 实机通信库未安装 | 按 [6.1](#61-前置条件) 安装 |
| 容器里有 `[python] <defunct>` 僵尸进程 | 父进程已退出留下的僵尸 | 不影响使用；重启容器可清理 |
| 真机程序无响应/机器人异常 | 未进入调试模式或网络不通 | 检查零力矩、`L2+R2` 调试模式、网卡配置、急停 |

---

## 10. 推荐工作流

### 10.1 标准流程

```bash
# 1) 冒烟测试：确认环境、显存、日志都正常
python legged_gym/scripts/train.py --task=g1 --headless \
  --num_envs=512 --max_iterations=10

# 2) 正式训练（可后台）
python legged_gym/scripts/train.py --task=g1 --headless

# 3) 另开终端启动 TensorBoard，观察 reward / episode_length
#    http://<容器IP>:6006

# 4) 训练过程中定期抽查策略（指定最近的 checkpoint）
python legged_gym/scripts/play.py --task=g1 --num_envs=8 \
  --load_run=<run目录名> --checkpoint=500

# 5) 训练结束后，用最终模型做 MuJoCo 验证
python legged_gym/scripts/play.py --task=g1 --num_envs=8
# 修改 deploy/deploy_mujoco/configs/g1.yaml 的 policy_path 后：
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# 6) 真机测试（确保安全和调试模式）
python deploy/deploy_real/deploy_real.py <网卡名> g1.yaml
```

### 10.2 目录约定建议

- 训练日志：`logs/g1/`
- 后台训练日志：`logs/train_g1.log`
- 导出的策略：`logs/g1/exported/policies/`
- 重要模型及时备份到宿主机其他目录

---

## 附录：G1 关键配置速查

来源：`legged_gym/envs/g1/g1_config.py`

| 项目 | 值 |
|---|---|
| 动作数 | 12（双腿） |
| 观测数 | 47 |
| 特权观测数 | 50 |
| 默认关节角 | hip_pitch -0.1、knee 0.3、ankle_pitch -0.2，其余 0 |
| stiffness | hip 100、knee 150、ankle 40 |
| damping | hip 2、knee 4、ankle 2 |
| `action_scale` | 0.25 |
| `decimation` | 4（控制频率 50 Hz） |
| 仿真步长 `dt` | 0.005 s |
| 回合时长 | 20 s（约 1000 个控制步） |
| 并行环境数 | 4096 |
| 最大迭代次数 | 10000 |
| 策略网络 | LSTM（隐藏层 64）+ MLP（32） |

其他机型只需把 `--task=g1` 换成 `--task=go2`、`--task=h1`、`--task=h1_2`，日志目录也会随之变为 `logs/<task>/`。

---

## 参考

- 项目说明：[README_zh.md](README_zh.md)
- 安装文档：[doc/setup_zh.md](doc/setup_zh.md)
- 实机部署说明：[deploy/deploy_real/README.zh.md](deploy/deploy_real/README.zh.md)
- 本地教程（同工作区 `src` 目录下）：
  - `../宇树G1强化学习教程（一）训练流程.pdf`
  - `../宇树G1强化学习教程（二）docker搭建.pdf`
- legged_gym: <https://github.com/leggedrobotics/legged_gym>
- rsl_rl: <https://github.com/leggedrobotics/rsl_rl>
- MuJoCo: <https://github.com/google-deepmind/mujoco>
- unitree_sdk2_python: <https://github.com/unitreerobotics/unitree_sdk2_python>

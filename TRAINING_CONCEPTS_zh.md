# G1 强化学习训练核心概念

> 以本仓库 G1 训练为例，凝练说明网络结构、迭代流程、Actor-Critic/PPO、Episode 等关键概念。
> 配置来源：[g1_config.py](legged_gym/envs/g1/g1_config.py)、[legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)
> 算法来源：`../rsl_rl/rsl_rl/`（仓库外的同级目录）

---

## 1. 术语与时间尺度

| 概念 | 定义 | G1 的值 |
|---|---|---|
| **策略步（policy step）** | 策略推理一次 + `env.step()` 一次 | 0.02 s |
| **物理步（sim step）** | Isaac Gym 推进一次物理仿真 | 0.005 s |
| **控制降频（decimation）** | 一个策略步包含的物理步数 | 4 |
| **transition** | 一个策略步产生的 `(obs, action, reward, done)` | 98,304 条/迭代 |
| **iteration（迭代）** | 一轮“采样 + PPO 更新” | 10,000 次 |
| **episode（回合）** | 一个机器人从 reset 到摔倒/超时 | 最长 1000 步 / 20 s |

频率关系：

```
物理/PD：200 Hz（0.005 s）
策略：  50 Hz （0.005 × 4 = 0.02 s）
回合：  20 s = 1000 个策略步
迭代：  每个环境采 24 步 = 0.48 s
```

---

## 2. 神经网络结构

配置在 [g1_config.py:91](legged_gym/envs/g1/g1_config.py#L91)：

```python
policy_class_name = "ActorCriticRecurrent"
actor_hidden_dims = [32]
critic_hidden_dims = [32]
activation = 'elu'
rnn_type = 'lstm'
rnn_hidden_size = 64
rnn_num_layers = 1
init_noise_std = 0.8
```

### 2.1 Actor（部署只用它）

```
obs_t (47)
   │
   ▼
LSTM(47 → 64, 1 层)  ◄──── h_{t-1}, c_{t-1}
   │ h_t (64)
   ▼
Linear(64 → 32) → ELU → Linear(32 → 12)
   │
   ▼
动作均值 μ_t (12)
```

- 训练时动作从 `Normal(μ, σ)` 采样，`σ` 是独立的 12 维可学习参数，初始 0.8
- 推理/部署时直接输出 μ（确定性）
- 输出经 `action_scale=0.25` 转成目标关节角偏移，再交给 PD 控制

### 2.2 Critic（仅训练使用）

```
privileged_obs_t (50 = 47 + 基座线速度)
   │
   ▼
LSTM(50 → 64) → Linear(64 → 32) → ELU → Linear(32 → 1)
   │
   ▼
状态价值 V(s)
```

非对称 Actor-Critic：Actor 不含基座线速度，Critic 含，训练更稳，部署时丢弃 Critic。

### 2.3 LSTM 的输入输出

```python
nn.LSTM(input_size=47, hidden_size=64, num_layers=1, batch_first=False)
```

| 项目 | 内容 |
|---|---|
| 输入 | 当前 47 维观测 + 上一时刻 `(h, c)`，各 64 维 |
| 输出 | 新的 `h_t` (64) 和 `c_t` (64) |
| 参数 | `weight_ih (256,47)`、`weight_hh (256,64)`、两个 bias (256) |
| 门顺序 | PyTorch：input / forget / cell / output |
| 参数数量 | Actor LSTM 28,928；Critic LSTM 29,696 |

不同阶段的张量形状：

| 阶段 | LSTM 输入 | 隐状态 |
|---|---|---|
| 采样 | `[1, 4096, 47]` | `(1, 4096, 64)` |
| PPO 更新 | `[T, B, 47]`（变长轨迹 padding） | `(1, B, 64)` |
| 部署 | `[1, 1, 47]` | `(1, 1, 64)` |

环境 `done` 时对应环境的 `h/c` 清零：

```python
hidden_state[..., dones, :] = 0.0
```

### 2.4 参数量与部署文件

| 部分 | 参数量 |
|---|---|
| Actor LSTM | 28,928 |
| Actor MLP | 2,476 |
| Actor std | 12 |
| **Actor 合计** | **31,416** |
| Critic LSTM | 29,696 |
| Critic MLP | 2,113 |
| **Critic 合计** | **31,809** |
| **总计** | **63,225** |

- 训练 checkpoint `model_*.pt`：约 760 KB（含优化器状态）
- 部署文件 `policy_lstm_1.pt`：约 143 KB（只含 Actor，输入 `[1,47]`，输出 `[1,12]`）

---

## 3. 每轮迭代的流程

核心参数：

```python
num_steps_per_env = 24
num_envs = 4096
num_learning_epochs = 5
num_mini_batches = 4
```

### 3.1 单次迭代时间线（单个环境）

```
step 1:  obs_0 → policy → action_0
         env.step: PD × 4 个物理步 (0.02 s) → obs_1, reward_1, done_1
step 2:  obs_1 → policy → action_1
         ...
step 24: obs_23 → policy → action_24
         → obs_24, reward_24, done_24

结束后：
  1. Critic 估计 V(obs_24) 作为 bootstrap
  2. 沿 24 步反向计算 GAE 和 returns
  3. PPO 更新：5 epoch × 4 mini-batch = 20 次梯度更新
  4. storage.clear() 丢弃数据
```

### 3.2 样本量

```
每轮迭代 = 24 步 × 4096 环境 = 98,304 个 transition
每步仿真时间 = 0.02 s
每轮覆盖仿真时间 = 0.48 s/环境
```

10,000 次迭代：

```
总 transition = 98,304 × 10,000 = 983,040,000
每环境策略步 = 240,000（约 80 分钟仿真时间）
估算完成回合 ≈ 190 万个（所有环境合计）
```

### 3.3 一个策略步内部

```
action (12)
  → clip
  → 目标关节角 = action × 0.25 + default_angle
  → PD: τ = Kp(q_target − q) − Kd·dq
  → 力矩 clip
  → 物理仿真 4 × 0.005 s
  → 计算 reward / obs / done
```

- 策略决策：50 Hz
- PD 力矩与物理：200 Hz（同一个动作目标保持 4 个物理步）

---

## 4. Actor-Critic 与 PPO 的关系

### 4.1 三个维度的区分

| 概念 | 回答的问题 |
|---|---|
| **Actor-Critic** | 网络怎么组成：策略网络 + 价值网络 |
| **PPO** | 用什么目标函数更新 |
| **On-policy** | 数据由哪个策略产生 |

**PPO 是一个 on-policy 的 Actor-Critic 算法。**

代码中体现为：

```python
policy_class_name = "ActorCriticRecurrent"   # 网络
algorithm_class_name = "PPO"                 # 算法
alg = PPO(actor_critic, ...)
```

### 4.2 为什么是 on-policy

```
采样（当前策略）→ 算 GAE → PPO 更新 → storage.clear() → 重新采样
```

- 没有 replay buffer
- 数据更新完立即丢弃
- 多个 epoch 复用同一批数据，但用重要性采样比 + clip 限制偏离

### 4.3 Actor 和 Critic 同时更新

一个 Adam 优化器覆盖全部参数：

```python
self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-3)
```

每个 mini-batch：

```python
surrogate_loss = ...                    # Actor
value_loss = ...                        # Critic
loss = surrogate_loss + 1.0 * value_loss - 0.01 * entropy

optimizer.zero_grad()
loss.backward()
clip_grad_norm_(actor_critic.parameters(), 1.0)
optimizer.step()
```

- 不是轮流更新
- Actor/Critic 参数不共享，梯度独立
- 一次反向传播同时更新两者

### 4.4 关键公式

```python
# 优势（GAE）
δ_t = r_t + γV(s_{t+1}) − V(s_t)
A_t = Σ (γλ)^k δ_{t+k}

# PPO 策略损失
ratio = exp(logπ_new − logπ_old)
surrogate = max(−A·ratio, −A·clip(ratio, 0.8, 1.2))

# 总损失
loss = surrogate + value_loss_coef · value_loss − entropy_coef · entropy
```

G1 超参：

| 参数 | 值 |
|---|---|
| γ | 0.99 |
| λ | 0.95 |
| clip | 0.2 |
| learning rate | 1e-3（adaptive，随 KL 调整） |
| value_loss_coef | 1.0 |
| entropy_coef | 0.01 |
| epochs × mini-batches | 5 × 4 |

---

## 5. Episode 的作用

### 5.1 定义

- 一个机器人从 reset 到摔倒或超时
- G1 最长 20 s = 1000 个策略步
- 一个回合约横跨 42 次迭代（1000 / 24）
- 回合中途摔倒时，该环境立即 reset，采样继续

### 5.2 作用

| 作用 | 说明 |
|---|---|
| 任务边界 | 摔倒/超时是明确的终止条件 |
| 正确的价值 bootstrap | 真终止不 bootstrap；超时用 `γV(s)` 截断 |
| 重置与多样性 | 随机初始状态、重新采样指令、清零 LSTM 记忆 |
| 统计评估 | `mean_reward`、`mean_episode_length` 都是回合级指标 |

终止处理（[ppo.py:104](file:///home/ccz/docker_catkin_ws/unitree_RL_tutorial/src/rsl_rl/rsl_rl/algorithms/ppo.py#L104)）：

```python
if 'time_outs' in infos:
    rewards += gamma * values * time_outs   # 超时截断，只做 bootstrap
```

- 摔倒：真终止，下一步价值为 0
- 超时：不是失败，用 `γV(s)` 补偿

### 5.3 回合不等于迭代

```
迭代：训练记账单位（24 步 × 4096 环境 + 一次 PPO 更新）
回合：任务单位（reset → done，最长 1000 步）
```

回合长度影响**任务难度和行为时间尺度**，不是直接决定价值精度。价值精度来自 Critic + GAE + bootstrap。

### 5.4 回合与 LSTM

回合结束时清零隐状态，避免跨回合污染：

```python
memory_a.reset(dones)
memory_c.reset(dones)
```

---

## 6. G1 关键参数速查

| 参数 | 值 |
|---|---|
| 观测（Actor） | 47 |
| 特权观测（Critic） | 50 |
| 动作 | 12 |
| 并行环境 | 4096 |
| 策略频率 | 50 Hz |
| 物理频率 | 200 Hz |
| 回合长度 | 20 s / 1000 步 |
| 每环境每轮采样 | 24 步 |
| PPO epochs × minibatches | 5 × 4 |
| 最大迭代 | 10,000 |
| 保存间隔 | 50 次迭代 |
| 网络 | LSTM 64 + MLP 32 |
| 总参数量 | 63,225 |
| 单轮 transition | 98,304 |

---

## 7. TensorBoard 指标速查

| 指标 | 含义 | 正常趋势 |
|---|---|---|
| `Train/mean_reward` | 最近 ≤100 个回合的平均总回报 | 上升 |
| `Train/mean_episode_length` | 最近 ≤100 个回合的平均步数 | 上升，接近 1000 |
| `Episode/rew_*` | 各奖励项的每秒平均值 | 正奖励上升；惩罚随运动增大而绝对值上升 |
| `Loss/value_function` | Critic 价值损失 | 小幅波动 |
| `Loss/surrogate` | PPO 代理损失 | 0 附近波动 |
| `Loss/learning_rate` | 自适应学习率 | 1e-3 附近 |
| `Policy/mean_noise_std` | 动作标准差均值 | 从 0.8 缓慢下降 |
| `Perf/total_fps` | 环境步/秒 | 本机约 8 万 |
| `Perf/collection time` | 采样耗时 | 约 1 s/迭代 |
| `Perf/learning_time` | 更新耗时 | 约 0.2 s/迭代 |

本次训练结果（10,000 迭代）：

```
mean_reward:         0.02 → 21.77
mean_episode_length: 15   → 1002
训练时长:            约 3 小时 6 分钟
显存:                约 5.7 GB / 8.2 GB
```

---

## 8. 一句话总结

> **策略每 0.02 秒输出一次动作；每轮迭代每个环境采 24 步，共 98,304 条数据，用 GAE 计算优势后做一次 PPO 更新（20 次梯度步）并丢弃数据；回合是机器人从 reset 到摔倒/超时的任务边界，用于重置环境、处理终止/截断和统计表现；LSTM 提供跨步记忆，在回合结束时清零。**

# Tianshou 和 Gym 使用指南

## 目录
- [1. Gym (OpenAI Gym) - 强化学习环境标准接口](#1-gym-openai-gym---强化学习环境标准接口)
- [2. Tianshou - 深度强化学习训练框架](#2-tianshou---深度强化学习训练框架)
- [3. RecordVideo 包装器详解](#3-recordvideo-包装器详解)
- [4. MuJoCo 集成指南](#4-mujoco-集成指南)
- [5. 实战示例](#5-实战示例)

---

## 1. Gym (OpenAI Gym) - 强化学习环境标准接口

### 1.1 什么是 Gym？

OpenAI Gym 是一个用于开发和比较强化学习算法的工具包，它提供了：
- **标准化的 API 接口**：`reset()`, `step()`, `render()`, `close()`
- **观察空间定义**：`observation_space` - 描述状态的维度和范围
- **动作空间定义**：`action_space` - 描述可执行的动作

### 1.2 CalliEnv 环境定义

在本项目中，我们自定义了书法环境 `CalliEnv`：

```python
class CalliEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, tool, folder_path, ...):
        # 定义动作空间：2维连续动作
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # 定义观察空间：8维状态
        self.observation_space = spaces.Box(
            low=np.array([0, r_min, l_min, theta_min, 0, 0, -1, -1]),
            high=np.array([1, r_max, l_max, theta_max, 1, 1, 1, 1]),
            dtype=np.float32
        )
```

### 1.3 状态空间（Observation Space）

CalliEnv 的状态由 **8 维向量**组成：

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `period` | [0, 1] | 当前笔画进度（position / stroke_length） |
| 1 | `r` | [0, 1] | 毛笔半径（归一化） |
| 2 | `l` | [0, 1] | 毛笔长度（归一化） |
| 3 | `theta` | [0, 1] | 毛笔旋转角度（归一化） |
| 4 | `curvature` | [0, 1] | 当前曲率（通过 sin 函数计算） |
| 5 | `r_prime` | [0, 1] | 上一步移动距离 |
| 6 | `vec_x` | [-1, 1] | 未来方向向量 X 分量 |
| 7 | `vec_y` | [-1, 1] | 未来方向向量 Y 分量 |

**示例状态：**
```python
state = [
    0.5,    # 笔画进行到一半
    0.6,    # 毛笔半径中等
    0.7,    # 毛笔长度较长
    0.3,    # 旋转角度约 54°
    0.2,    # 曲率较低（接近直线）
    0.1,    # 上一步移动距离小
    0.8,    # 向右上方移动
    0.6
]
```

### 1.4 动作空间（Action Space）

CalliEnv 的动作由 **2 维连续向量**组成：

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `r_prime` | [-1, 1] | 移动距离（极坐标），实际距离 = r_prime × 0.022 |
| 1 | `theta_prime` | [-1, 1] | 移动角度（极坐标），实际角度 = theta_prime × π |

**动作示例：**
```python
action = [0.5, 0.3]  # 移动距离 0.011m，角度 0.3π ≈ 54°
```

### 1.5 奖励函数（Reward Function）

```python
reward = -2 * abs(r_prime_new) * curvature / 0.4 * cos_sim(theta, new_theta) + 0.6
```

**奖励设计原则：**
- ❌ **惩罚大的移动距离**：鼓励平滑、小步幅的运动
- ❌ **惩罚高曲率区域的大移动**：在曲线处需要更小心
- ✅ **奖励方向一致性**：运动方向与目标方向的余弦相似度越高越好

### 1.6 环境核心方法

```python
class CalliEnv(gym.Env):
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        # 1. 重置画布
        self.canvas = np.ones((256, 256, 3), dtype=np.uint8) * 255

        # 2. 加载新的目标图像和骨架
        self.current_image = self._load_random_image()
        self.skeleton = self._extract_skeleton(self.current_image)

        # 3. 重置笔画进度
        self.stroke_index = 0
        self.point_index = 0

        # 4. 返回初始观察
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """执行一个动作"""
        # 1. 解析动作
        r_prime = action[0] * self.r_prime_bound  # [-0.022, 0.022]
        theta_prime = action[1] * np.pi           # [-π, π]

        # 2. 计算新位置（极坐标转换）
        dx = r_prime * np.cos(theta_prime)
        dy = r_prime * np.sin(theta_prime)
        new_pos = self.current_pos + np.array([dx, dy])

        # 3. 自动拟合毛笔参数（r, l, theta）
        r_new, l_new, theta_new = self.tool.auto_fit(
            new_pos, self.skeleton, self.current_image
        )

        # 4. 绘制笔画
        self._draw_stroke(self.current_pos, new_pos, r_new, l_new, theta_new)

        # 5. 计算奖励
        reward = self._compute_reward(r_prime, theta_new, curvature)

        # 6. 更新状态
        self.current_pos = new_pos
        self.point_index += 1

        # 7. 检查是否完成
        done = (self.point_index >= len(self.skeleton))
        truncated = False

        # 8. 返回新观察
        obs = self._get_observation()
        info = {'canvas': self.canvas.copy()}

        return obs, reward, done, truncated, info

    def render(self):
        """渲染当前状态"""
        if self.render_mode == 'rgb_array':
            # 返回 numpy 数组 (H, W, 3)
            return self.canvas.copy()

        elif self.render_mode == 'human':
            # 显示窗口
            cv2.imshow('CalliEnv', self.canvas)
            cv2.waitKey(1)
            return None
```

---

## 2. Tianshou - 深度强化学习训练框架

### 2.1 什么是 Tianshou？

Tianshou（天授）是清华大学开发的深度强化学习库，特点：
- ✅ **模块化设计**：Policy, Collector, Buffer, Trainer 各司其职
- ✅ **高效并行**：支持多进程环境和数据收集
- ✅ **算法丰富**：DQN, PPO, SAC, DDPG, TD3 等
- ✅ **易于扩展**：方便实现自定义策略和环境

官方文档：https://tianshou.readthedocs.io/

### 2.2 核心组件

#### 2.2.1 环境包装器（Environment Wrappers）

```python
from tianshou.env import SubprocVectorEnv, DummyVectorEnv

# 训练环境 - 多进程并行（速度快）
train_envs = SubprocVectorEnv([
    lambda i=i: gym.make('CalliEnv-v0',
                         folder_path=train_data_dir,
                         env_rank=(i*100, (i+1)*100),  # 每个进程不同的数据
                         render_mode='rgb_array')
    for i in range(4)  # 4 个并行环境
])

# 测试环境 - 单进程串行（稳定可靠）
test_envs = DummyVectorEnv([
    lambda i=i: gym.make('CalliEnv-v0',
                         folder_path=test_data_dir,
                         render_mode='rgb_array')
    for i in range(4)
])
```

**对比：**

| 特性 | SubprocVectorEnv | DummyVectorEnv |
|------|------------------|----------------|
| 并行方式 | 多进程（真并行） | 单进程（串行） |
| 速度 | 快（4倍加速） | 慢 |
| 内存 | 每个进程独立内存 | 共享内存 |
| 调试 | 困难（多进程） | 容易 |
| 适用 | 训练阶段 | 测试/调试 |

#### 2.2.2 策略网络（Policy Network）

**Actor-Critic 架构：**

```python
from tianshou.utils.net.continuous import ActorProb, Critic
from MLP.model import My_MLP

# Actor 网络（策略网络）- 输出动作分布
actor_net = My_MLP(
    state_dim=8,              # 输入：8维状态
    hidden_sizes=(128, 256),  # 隐藏层
    learn_fourier=True,       # 傅里叶特征编码
    fourier_dim=256
)
actor = ActorProb(
    actor_net,
    action_dim=2,             # 输出：2维动作
    hidden_sizes=(64,)        # μ 和 σ 的网络
)

# Critic 网络（价值网络）- 评估状态-动作对的价值
critic_net = My_MLP(
    state_dim + action_dim,   # 输入：状态 + 动作
    hidden_sizes=(128, 256, 256),
    concat=True
)
critic_1 = Critic(critic_net)
critic_2 = Critic(critic_net)  # 双 Critic（SAC 特点）
```

**网络结构可视化：**

```
State (8维)
    ↓
[傅里叶特征编码] (可选)
    ↓
State' (8 + 256维)
    ↓
Linear(128) + ReLU
    ↓
Linear(256) + ReLU
    ↓
┌─────────────┬─────────────┐
│   μ (mean)  │  σ (std)    │  ← Actor 输出
└─────────────┴─────────────┘
      ↓
Gaussian Distribution
      ↓
   Action (2维)
```

#### 2.2.3 SAC 策略（Soft Actor-Critic）

```python
from tianshou.policy import SACPolicy

# 熵正则化系数（自动调整）
target_entropy = 0.98 * torch.log(torch.tensor(float(action_dim)))
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

# SAC 策略
policy = SACPolicy(
    actor=actor,
    actor_optim=torch.optim.Adam(actor.parameters(), lr=3e-5),
    critic1=critic_1,
    critic1_optim=torch.optim.Adam(critic_1.parameters(), lr=1e-4),
    critic2=critic_2,
    critic2_optim=torch.optim.Adam(critic_2.parameters(), lr=1e-4),
    tau=0.005,                    # 软更新系数
    gamma=0.9,                    # 折扣因子
    alpha=(target_entropy, log_alpha, alpha_optim),
    estimation_step=1,            # n-step return
    action_space=env.action_space
)
```

**SAC 算法特点：**
- ✅ **Off-policy**：可以使用旧数据训练（样本效率高）
- ✅ **最大熵**：鼓励探索，避免过早收敛
- ✅ **双 Critic**：减少价值过估计（double Q-learning）
- ✅ **自动调整熵系数**：平衡探索与利用

**SAC 更新公式：**

```
Critic Loss:
L(Q) = E[(Q(s,a) - (r + γ * (Q'(s',a') - α*log π(a'|s'))))²]

Actor Loss:
L(π) = E[α*log π(a|s) - Q(s,a)]

Alpha Loss:
L(α) = E[-α * (log π(a|s) + H_target)]
```

#### 2.2.4 经验回放缓冲区（Replay Buffer）

```python
from tianshou.data import VectorReplayBuffer

buffer = VectorReplayBuffer(
    total_size=2**20,      # 总容量：1,048,576 条 transition
    buffer_num=4           # 4 个并行环境
)
```

**Buffer 存储格式：**

```python
transition = {
    'obs': state,          # 当前状态 (8,)
    'act': action,         # 执行的动作 (2,)
    'rew': reward,         # 获得的奖励 (1,)
    'done': done,          # 是否结束 (bool)
    'obs_next': next_state,# 下一个状态 (8,)
    'info': {...}          # 额外信息
}
```

**采样机制：**
```python
# 从 buffer 中随机采样 batch
batch = buffer.sample(batch_size=2048)
# batch.obs: (2048, 8)
# batch.act: (2048, 2)
# batch.rew: (2048, 1)
```

#### 2.2.5 数据收集器（Collector）

```python
from tianshou.data import Collector

# 训练数据收集器
train_collector = Collector(
    policy=policy,
    env=train_envs,
    buffer=buffer,
    exploration_noise=True  # 添加探索噪声
)

# 预填充 buffer（使用随机策略）
train_collector.collect(n_step=2048, random=True)

# 测试数据收集器
test_collector = Collector(
    policy=policy,
    env=test_envs,
    exploration_noise=False  # 测试时不探索
)
```

**收集流程：**

```
1. policy.forward(obs) → action
2. env.step(action) → (next_obs, reward, done, info)
3. buffer.add(obs, action, reward, next_obs, done)
4. 重复直到收集 n_step 个 transition
```

#### 2.2.6 训练器（Trainer）

```python
from tianshou.trainer import offpolicy_trainer

result = offpolicy_trainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,

    # 训练配置
    max_epoch=150,              # 最大训练轮数
    step_per_epoch=10000,       # 每轮最多收集 10000 步
    step_per_collect=16,        # 每 16 步更新一次网络
    update_per_step=2.0,        # 每次收集后更新 2 次

    # 测试配置
    episode_per_test=10,        # 每次测试 10 个 episode
    batch_size=2048,            # 每次更新的 batch 大小

    # 其他
    logger=logger,              # TensorBoard 日志
    verbose=True,               # 打印详细信息
    show_progress=True,         # 显示进度条
    test_in_train=True          # 训练中定期测试
)
```

**训练循环详解：**

```python
for epoch in range(max_epoch):
    # 1. 训练阶段
    collected_steps = 0
    while collected_steps < step_per_epoch:
        # 收集数据
        result = train_collector.collect(n_step=step_per_collect)
        collected_steps += result['n/st']

        # 更新网络
        for _ in range(int(step_per_collect * update_per_step)):
            batch = buffer.sample(batch_size)
            losses = policy.learn(batch)

    # 2. 测试阶段
    test_result = test_collector.collect(n_episode=episode_per_test)
    mean_reward = test_result['rew']

    # 3. 日志记录
    logger.log_train_data({'loss/actor': losses.actor_loss}, epoch)
    logger.log_test_data({'reward/mean': mean_reward}, epoch)

    # 4. 保存最佳模型
    if mean_reward > best_reward:
        torch.save(policy.state_dict(), 'best_policy.pth')
```

### 2.3 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     Epoch 循环 (150 轮)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │         训练阶段 (10000 步)            │
        └───────────────────────────────────────┘
                    ↓                   ↑
        ┌─────────────────────┐        │
        │  收集数据 (16 步)    │        │
        │  4个环境并行          │        │
        └─────────────────────┘        │
                    ↓                   │
        ┌─────────────────────┐        │
        │  存入 Buffer         │        │
        └─────────────────────┘        │
                    ↓                   │
        ┌─────────────────────┐        │
        │  更新网络 (2 次)     │        │
        │  • 采样 2048 个样本  │        │
        │  • 计算 loss         │        │
        │  • 梯度下降          │        │
        └─────────────────────┘        │
                    ↓                   │
        ┌─────────────────────┐        │
        │  未达到 10000 步？   ├────────┘
        └─────────────────────┘
                    ↓
        ┌───────────────────────────────────────┐
        │         测试阶段 (10 episodes)         │
        │  • 不使用探索噪声                      │
        │  • 记录平均奖励                        │
        │  • 保存最佳模型                        │
        └───────────────────────────────────────┘
```

### 2.4 并行训练架构

```
                    ┌──────────────────────┐
                    │    主进程（训练）      │
                    │  ┌────────────────┐  │
                    │  │ Policy Network │  │
                    │  └────────────────┘  │
                    │          ↓           │
                    │  ┌────────────────┐  │
                    │  │ Replay Buffer  │  │
                    │  │   1M samples   │  │
                    │  └────────────────┘  │
                    └──────────────────────┘
                              ↑
                    收集 transitions
                              │
        ┌─────────┬───────────┼───────────┬─────────┐
        │         │           │           │         │
    ┌───┴───┐ ┌──┴───┐  ┌───┴───┐  ┌───┴───┐ ┌───┴───┐
    │ Env 0 │ │ Env 1│  │ Env 2 │  │ Env 3 │ │  ...  │
    │(进程0)│ │(进程1)│ │(进程2)│ │(进程3)│ │       │
    └───┬───┘ └──┬───┘  └───┬───┘  └───┬───┘ └───────┘
        │        │          │          │
    ┌───┴───┐ ┌──┴───┐  ┌───┴───┐  ┌───┴───┐
    │图像0-99│ │100-199│ │200-299│ │300-399│  不同数据集
    └────────┘ └───────┘ └────────┘ └────────┘
```

**并行加速比：**
- 4 个环境 → **约 3.5x 加速**（考虑通信开销）
- 8 个环境 → **约 6x 加速**
- 16 个环境 → **约 10x 加速**（受限于 CPU 核心数）

---

## 3. RecordVideo 包装器详解

### 3.1 什么是 RecordVideo？

`RecordVideo` 是 Gym 提供的视频录制包装器，它可以：
- ✅ 自动录制环境的每一帧
- ✅ 在 episode 结束时保存为视频
- ✅ 灵活控制录制时机
- ✅ 支持并行环境

### 3.2 工作原理

```python
from gym.wrappers.record_video import RecordVideo

class RecordVideo(gym.Wrapper):
    """视频录制包装器（简化版）"""

    def __init__(self, env, video_folder, name_prefix='video',
                 episode_trigger=None, step_trigger=None):
        super().__init__(env)
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.episode_trigger = episode_trigger or (lambda x: True)
        self.step_trigger = step_trigger

        self.frames = []
        self.recording = False
        self.episode_count = 0

    def reset(self, **kwargs):
        # 1. 检查是否应该开始录制
        if self.episode_trigger(self.episode_count):
            self.recording = True
            self.frames = []
            print(f"📹 开始录制 Episode {self.episode_count}")

        # 2. 调用原环境的 reset
        obs = self.env.reset(**kwargs)

        # 3. 如果正在录制，获取第一帧
        if self.recording:
            frame = self.env.render()  # ← 关键！调用 render()
            if frame is not None:
                self.frames.append(frame)

        return obs

    def step(self, action):
        # 1. 执行原环境的 step
        obs, reward, done, truncated, info = self.env.step(action)

        # 2. 如果正在录制，获取当前帧
        if self.recording:
            if self.step_trigger is None or self.step_trigger(self.step_count):
                frame = self.env.render()  # ← 关键！调用 render()
                if frame is not None:
                    self.frames.append(frame)

        # 3. 如果 episode 结束，保存视频
        if (done or truncated) and self.recording:
            self._save_video()
            self.recording = False
            self.episode_count += 1

        return obs, reward, done, truncated, info

    def _save_video(self):
        """保存视频文件"""
        import imageio

        if len(self.frames) == 0:
            return

        # 生成文件名
        filename = f"{self.video_folder}/{self.name_prefix}_episode_{self.episode_count}.mp4"

        # 保存视频
        imageio.mimsave(filename, self.frames, fps=30)
        print(f"✅ 视频已保存: {filename} ({len(self.frames)} 帧)")

        # 清空缓存
        self.frames = []
```

### 3.3 使用示例

#### 基础用法

```python
import gym
from gym.wrappers.record_video import RecordVideo

# 创建环境
env = gym.make('CalliEnv-v0', render_mode='rgb_array')

# 包装 RecordVideo
env = RecordVideo(
    env,
    video_folder='./videos',
    name_prefix='calligraphy',
    episode_trigger=lambda x: True,  # 每个 episode 都录
)

# 训练循环（会自动录制）
for episode in range(10):
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        # RecordVideo 会在后台自动录制每一帧！

env.close()  # 确保最后一个视频被保存
```

**生成的视频文件：**
```
./videos/
├── calligraphy_episode_0.mp4
├── calligraphy_episode_1.mp4
├── calligraphy_episode_2.mp4
└── ...
```

#### 高级用法：条件录制

```python
# 只录制前 5 个 episode
env = RecordVideo(
    env,
    video_folder='./videos',
    episode_trigger=lambda x: x < 5
)

# 每 10 个 episode 录制一次
env = RecordVideo(
    env,
    video_folder='./videos',
    episode_trigger=lambda x: x % 10 == 0
)

# 只录制高分 episode（需要自定义）
class SelectiveRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, reward_threshold=100):
        super().__init__(env, video_folder, episode_trigger=lambda x: False)
        self.reward_threshold = reward_threshold
        self.episode_reward = 0

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.episode_reward += reward

        # 达到阈值才开始录制
        if self.episode_reward >= self.reward_threshold and not self.recording:
            self.recording = True
            print(f"📹 高分episode！开始录制（当前奖励: {self.episode_reward}）")

        return obs, reward, done, truncated, info
```

### 3.4 与 Tianshou 集成

在本项目中的实际使用：

```python
from tianshou.env import SubprocVectorEnv
from gym.wrappers.record_video import RecordVideo

# 4 个并行训练环境，每个都录制视频
train_envs = SubprocVectorEnv([
    lambda i=i: RecordVideo(
        gym.make('CalliEnv-v0',
                 tool=tool,
                 folder_path=args.train_data_dir,
                 env_rank=(i*100, (i+1)*100),  # 每个进程不同数据
                 render_mode='rgb_array'),     # ← 必须是 rgb_array
        video_folder=save_video_dir,
        name_prefix=f'trainvids_{i}',          # 区分不同进程
        new_step_api=True
    ) for i in range(4)
])
```

**并行录制效果：**
```
./videos/
├── trainvids_0_episode_0.mp4   # 进程 0 录制
├── trainvids_0_episode_1.mp4
├── trainvids_1_episode_0.mp4   # 进程 1 录制
├── trainvids_1_episode_1.mp4
├── trainvids_2_episode_0.mp4   # 进程 2 录制
├── trainvids_3_episode_0.mp4   # 进程 3 录制
└── ...
```

---

## 4. MuJoCo 集成指南

### 4.1 MuJoCo 环境的 render() 实现

MuJoCo 完全支持 Gym 的 render 接口：

```python
import mujoco
import numpy as np

class MuJoCoCalligraphyEnv:
    def __init__(self, render_mode='rgb_array'):
        self.render_mode = render_mode

        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path('franka_fr3v2_calligraphy.xml')
        self.data = mujoco.MjData(self.model)

        # 创建渲染器（仅在 rgb_array 模式下）
        if render_mode == 'rgb_array':
            self.renderer = mujoco.Renderer(
                self.model,
                height=720,
                width=1280
            )
            self.camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                'top_view'
            )

    def render(self):
        """渲染当前状态"""
        if self.render_mode == 'rgb_array':
            # 更新场景
            self.renderer.update_scene(self.data, camera=self.camera_id)

            # 渲染并返回 RGB 数组
            frame = self.renderer.render()  # 返回 (720, 1280, 3) uint8
            return frame

        elif self.render_mode == 'human':
            # 使用 MuJoCo 的交互式 viewer
            if not hasattr(self, 'viewer'):
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
            return None

        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")
```

### 4.2 RecordVideo 可以直接用于 MuJoCo 吗？

**✅ 完全可以！**

只要你的 MuJoCo 环境：
1. 继承 `gym.Env`
2. 实现 `render(mode='rgb_array')` 返回 `(H, W, 3)` 的 numpy 数组
3. 在 `__init__` 中设置 `render_mode='rgb_array'`

**示例：**

```python
import gym
from gym import spaces
import mujoco
from gym.wrappers.record_video import RecordVideo

class FrankaCalligraphyGymEnv(gym.Env):
    """MuJoCo 书法环境（Gym 包装）"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode

        # 加载 MuJoCo 模型
        from mujoco_simulator import FrankaCalligraphySimulator
        self.sim = FrankaCalligraphySimulator(render_mode=render_mode)

        # 定义空间
        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 创建渲染器
        if render_mode == 'rgb_array':
            self.renderer = mujoco.Renderer(self.sim.model, 720, 1280)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 执行动作
        target_pos = self._action_to_target(action)
        self.sim.move_to_position(target_pos)

        # 获取新状态
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()

        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode == 'rgb_array':
            camera_id = 0  # top_view
            self.renderer.update_scene(self.sim.data, camera=camera_id)
            frame = self.renderer.render()  # ← 返回 (720, 1280, 3)
            return frame
        elif self.render_mode == 'human':
            if not hasattr(self, 'viewer'):
                self.viewer = mujoco.viewer.launch_passive(
                    self.sim.model, self.sim.data
                )
            self.viewer.sync()

    def _get_obs(self):
        qpos = self.sim.data.qpos[:7].copy()
        ee_pos = self.sim.data.site_xpos[...].copy()
        return np.concatenate([qpos, ee_pos])

    def _compute_reward(self):
        # 基于接触率、笔画质量等
        return -np.linalg.norm(self.sim.data.qvel[:7])

    def _check_done(self):
        return False  # 或根据轨迹完成情况


# 注册环境
from gym.envs.registration import register
register(
    id='FrankaCalligraphy-v0',
    entry_point='mujoco_gym_env:FrankaCalligraphyGymEnv',
)

# 使用 RecordVideo 包装
env = gym.make('FrankaCalligraphy-v0', render_mode='rgb_array')
env = RecordVideo(env, './mujoco_videos', 'franka')

# 正常使用（自动录制）
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        break

env.close()
# 视频自动保存到 ./mujoco_videos/franka_episode_0.mp4
```

### 4.3 对比：RecordVideo vs 手动录制

在你的项目中，你使用的是手动录制方法：

```python
# 当前方法：手动录制
frames = []
for i in range(num_points):
    sim.move_to_position(target_pos)

    # 手动渲染
    renderer.update_scene(sim.data, camera=camera_id)
    frame = renderer.render()
    frames.append(frame)

# 手动保存
imageio.mimsave('output.mp4', frames, fps=30)
```

**使用 RecordVideo 的优势：**

| 特性 | RecordVideo | 手动录制 |
|------|-------------|---------|
| 代码量 | 1 行包装 | ~10 行管理 |
| 内存管理 | 自动释放 | 手动管理 frames |
| 错误处理 | 自动处理 | 需要 try-catch |
| 与 RL 集成 | 无缝集成 | 需要自定义 |
| 并行支持 | 自动支持 | 需要手动同步 |
| 灵活性 | 中等 | 高 |

**建议：**
- �� **RL 训练监控** → 使用 RecordVideo
- 🎯 **展示视频生成** → 使用手动录制（更灵活，可以后处理）

---

## 5. 实战示例

### 5.1 完整的训练脚本

结合 Gym、Tianshou、RecordVideo 的完整示例：

```python
#!/usr/bin/env python3
"""
完整的 RL 训练脚本（书法任务）
使用 Gym + Tianshou + RecordVideo
"""

import gym
import torch
import numpy as np
from gym.wrappers.record_video import RecordVideo
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.utils.tensorboard import SummaryWriter

# ==================== 1. 环境设置 ====================

# 创建单个环境（用于获取空间信息）
env = gym.make('CalliEnv-v0',
               tool=tool,
               folder_path='./train_data',
               render_mode='rgb_array')

obs_shape = env.observation_space.shape[0]  # 8
action_shape = env.action_space.shape[0]    # 2

# 创建训练环境（4 个并行 + 视频录制）
train_envs = SubprocVectorEnv([
    lambda i=i: RecordVideo(
        gym.make('CalliEnv-v0',
                 tool=tool,
                 folder_path='./train_data',
                 env_rank=(i*100, (i+1)*100),
                 render_mode='rgb_array'),
        video_folder='./videos/train',
        name_prefix=f'train_{i}',
        episode_trigger=lambda x: x % 50 == 0  # 每 50 个 episode 录一次
    ) for i in range(4)
])

# 创建测试环境（2 个串行 + 视频录制）
test_envs = DummyVectorEnv([
    lambda i=i: RecordVideo(
        gym.make('CalliEnv-v0',
                 tool=tool,
                 folder_path='./test_data',
                 render_mode='rgb_array'),
        video_folder='./videos/test',
        name_prefix=f'test_{i}',
        episode_trigger=lambda x: True  # 每个 episode 都录
    ) for i in range(2)
])

# ==================== 2. 网络定义 ====================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Actor 网络
from MLP.model import My_MLP
actor_net = My_MLP(obs_shape, hidden_sizes=(128, 256), device=device).to(device)
actor = ActorProb(actor_net, action_shape, hidden_sizes=(64,), device=device).to(device)

# Critic 网络（双网络）
critic_net_1 = My_MLP(obs_shape + action_shape, hidden_sizes=(128, 256, 256),
                      concat=True, device=device).to(device)
critic_net_2 = My_MLP(obs_shape + action_shape, hidden_sizes=(128, 256, 256),
                      concat=True, device=device).to(device)
critic_1 = Critic(critic_net_1, device=device).to(device)
critic_2 = Critic(critic_net_2, device=device).to(device)

# 优化器
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-5)
critic1_optim = torch.optim.Adam(critic_1.parameters(), lr=1e-4)
critic2_optim = torch.optim.Adam(critic_2.parameters(), lr=1e-4)

# 熵正则化
target_entropy = 0.98 * torch.log(torch.tensor(float(action_shape)))
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

# ==================== 3. 策略定义 ====================

policy = SACPolicy(
    actor, actor_optim,
    critic_1, critic1_optim,
    critic_2, critic2_optim,
    tau=0.005,
    gamma=0.9,
    alpha=(target_entropy, log_alpha, alpha_optim),
    action_space=env.action_space
)

# ==================== 4. 数据收集 ====================

# Replay Buffer
buffer = VectorReplayBuffer(total_size=2**20, buffer_num=len(train_envs))

# Collector
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=False)

# 预填充
train_collector.collect(n_step=2048, random=True)

# ==================== 5. 训练 ====================

# Logger
writer = SummaryWriter('./logs')
logger = TensorboardLogger(writer)

# 开始训练
result = offpolicy_trainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=150,
    step_per_epoch=10000,
    step_per_collect=16,
    update_per_step=2.0,
    episode_per_test=10,
    batch_size=2048,
    logger=logger,
    verbose=True,
    show_progress=True,
    test_in_train=True
)

# ==================== 6. 保存和测试 ====================

# 保存模型
torch.save(policy.state_dict(), './models/sac_policy.pth')

# 加载并测试
policy.load_state_dict(torch.load('./models/sac_policy.pth'))
policy.eval()

# 最终测试（会自动录制视频）
test_result = test_collector.collect(n_episode=10, render=1/30)
print(f"测试平均奖励: {test_result['rew']:.2f}")
print(f"测试视频已保存到: ./videos/test/")

env.close()
train_envs.close()
test_envs.close()
```

### 5.2 MuJoCo 环境包装示例

将你的 MuJoCo 仿真器包装成 Gym 环境：

```python
# mujoco_gym_env.py
import gym
from gym import spaces
import numpy as np
import mujoco
from mujoco_simulator import FrankaCalligraphySimulator

class FrankaCalligraphyGymEnv(gym.Env):
    """Franka 书法 MuJoCo 环境（Gym 接口）"""

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, npz_path, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode

        # 初始化仿真器
        self.sim = FrankaCalligraphySimulator(render_mode=render_mode)

        # 加载轨迹
        data = np.load(npz_path)
        self.trajectory_x = data['pos_3d_x']
        self.trajectory_y = data['pos_3d_y']
        self.trajectory_z = data['pos_3d_z']
        self.num_points = len(self.trajectory_x)

        # 定义观察空间：[当前关节角度(7) + 目标位置(3) + 进度(1)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        # 定义动作空间：[目标关节角度增量(7)]
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(7,),
            dtype=np.float32
        )

        # 创建渲染器
        if render_mode == 'rgb_array':
            self.renderer = mujoco.Renderer(self.sim.model, height=720, width=1280)
            self.camera_id = mujoco.mj_name2id(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                'top_view'
            )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置仿真器
        self.sim.reset()
        self.current_step = 0

        # 获取初始观察
        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # 1. 应用动作（关节角度增量）
        current_qpos = self.sim.data.qpos[:7].copy()
        target_qpos = np.clip(current_qpos + action, -np.pi, np.pi)

        # 2. 执行控制
        self.sim.data.ctrl[:7] = target_qpos
        for _ in range(10):  # 仿真 10 步
            mujoco.mj_step(self.sim.model, self.sim.data)

        # 3. 获取新观察
        obs = self._get_obs()

        # 4. 计算奖励
        reward = self._compute_reward()

        # 5. 检查是否完成
        self.current_step += 1
        done = (self.current_step >= self.num_points)
        truncated = False

        # 6. 额外信息
        info = {
            'contact_rate': self._get_contact_rate(),
            'canvas': self.sim.paper_canvas.copy()
        }

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            # MuJoCo 渲染
            self.renderer.update_scene(self.sim.data, camera=self.camera_id)
            frame = self.renderer.render()  # (720, 1280, 3)

            # 叠加画布（可选）
            canvas_overlay = self._create_canvas_overlay(frame)
            return canvas_overlay

        elif self.render_mode == 'human':
            if not hasattr(self, 'viewer'):
                self.viewer = mujoco.viewer.launch_passive(
                    self.sim.model, self.sim.data
                )
            self.viewer.sync()
            return None

    def _get_obs(self):
        """获取当前观察"""
        # 当前关节角度
        qpos = self.sim.data.qpos[:7].copy()

        # 目标位置（下一个轨迹点）
        if self.current_step < self.num_points:
            target_x = self.trajectory_x[self.current_step]
            target_y = self.trajectory_y[self.current_step]
            target_z = self.trajectory_z[self.current_step]
        else:
            target_x = target_y = target_z = 0.0

        # 进度
        progress = self.current_step / self.num_points

        return np.concatenate([qpos, [target_x, target_y, target_z, progress]])

    def _compute_reward(self):
        """计算奖励"""
        # 获取末端执行器位置
        ee_site_id = mujoco.mj_name2id(
            self.sim.model,
            mujoco.mjtObj.mjOBJ_SITE,
            'ee_site'
        )
        ee_pos = self.sim.data.site_xpos[ee_site_id]

        # 目标位置
        target_pos = np.array([
            self.trajectory_x[self.current_step] + self.sim.paper_offset[0],
            self.trajectory_y[self.current_step] + self.sim.paper_offset[1],
            self.trajectory_z[self.current_step] + 0.011
        ])

        # 位置误差
        distance = np.linalg.norm(ee_pos - target_pos)
        position_reward = -distance * 10.0

        # 接触奖励
        brush_pos, contact_force = self.sim.get_brush_contact()
        contact_reward = 1.0 if contact_force > 0.0001 else -0.5

        # 速度惩罚（鼓励平滑运动）
        velocity_penalty = -0.01 * np.linalg.norm(self.sim.data.qvel[:7])

        return position_reward + contact_reward + velocity_penalty

    def _get_contact_rate(self):
        """计算接触率"""
        if len(self.sim.ink_traces) == 0:
            return 0.0
        contact_points = sum(1 for _, _, _, c in self.sim.ink_traces if c)
        return contact_points / len(self.sim.ink_traces)

    def _create_canvas_overlay(self, robot_frame):
        """在机器人视图上叠加画布"""
        import cv2

        composite = robot_frame.copy()
        canvas = self.sim.paper_canvas

        # 缩放画布
        h, w = robot_frame.shape[:2]
        canvas_w = int(w * 0.3)
        canvas_h = int(canvas_w * canvas.shape[0] / canvas.shape[1])
        canvas_resized = cv2.resize(canvas, (canvas_w, canvas_h))

        # 转换为 RGB
        canvas_rgb = cv2.cvtColor(canvas_resized, cv2.COLOR_GRAY2RGB)

        # 叠加到右下角
        x_offset = w - canvas_w - 10
        y_offset = h - canvas_h - 10
        composite[y_offset:y_offset+canvas_h, x_offset:x_offset+canvas_w] = canvas_rgb

        return composite

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
        self.sim.close()


# 注册环境
from gym.envs.registration import register
register(
    id='FrankaCalligraphy-v0',
    entry_point='mujoco_gym_env:FrankaCalligraphyGymEnv',
    max_episode_steps=1000,
)
```

**使用示例：**

```python
from gym.wrappers.record_video import RecordVideo

# 创建环境
env = gym.make(
    'FrankaCalligraphy-v0',
    npz_path='./demo_outputs/test_0_large.npz',
    render_mode='rgb_array'
)

# 包装 RecordVideo
env = RecordVideo(
    env,
    video_folder='./mujoco_rl_videos',
    name_prefix='franka_learning',
    episode_trigger=lambda x: True
)

# RL 训练（使用 Tianshou）
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy

# ... (省略网络和策略定义)

# 创建 Collector（会自动录制视频）
collector = Collector(policy, env, buffer)
collector.collect(n_episode=10)

# 视频自动保存到 ./mujoco_rl_videos/
```

---

## 总结

### 核心要点

1. **Gym** 提供标准化的 RL 环境接口
   - `reset()`, `step()`, `render()`, `close()`
   - 定义观察空间和动作空间

2. **Tianshou** 提供高效的 RL 训练框架
   - Policy, Collector, Buffer, Trainer 模块化设计
   - 支持并行环境和多种算法

3. **RecordVideo** 自动录制训练过程
   - 在每次 `step()` 后调用 `render()`
   - 自动保存为视频文件
   - 支持并行环境

4. **MuJoCo 完全兼容**
   - 只需实现 `render(mode='rgb_array')`
   - 返回 `(H, W, 3)` 的 numpy 数组
   - 可以直接使用 RecordVideo

### 最佳实践

✅ **使用 RecordVideo 的场景：**
- RL 训练监控
- 自动化测试
- 并行环境录制

✅ **手动录制的场景：**
- 需要后处理（叠加画布、添加文字等）
- 生成展示视频
- 需要完全控制录制时机

✅ **MuJoCo + Gym + Tianshou 组合：**
- 最强大的 RL 训练栈
- 高效并行 + 自动录制
- 适合复杂的机器人控制任务

---

## 参考资料

- **Gym 官方文档**: https://www.gymlibrary.dev/
- **Tianshou 官方文档**: https://tianshou.readthedocs.io/
- **MuJoCo 官方文档**: https://mujoco.readthedocs.io/
- **SAC 论文**: Soft Actor-Critic (Haarnoja et al., 2018)
- **本项目代码**: `/Users/seer/CalliRewrite/rl_finetune/`

---

**文档版本**: v1.0
**最后更新**: 2026-01-22
**作者**: Claude Code

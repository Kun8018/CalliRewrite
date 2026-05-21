# CalliRewrite 强化学习系统总览

## 📚 项目概述

CalliRewrite 是一个创新的书法重写系统，它使用强化学习（RL）技术将输入的书法图像转换为可执行的笔画序列。本项目的强化学习部分（rl_finetune）负责优化从序列提取阶段获得的粗笔画序列，使其适应特定书写工具的物理特性。

**核心目标**：将粗笔画序列转换为符合真实工具（毛笔、马克笔等）物理动力学的精细轨迹

## 🏗️ 整体架构

CalliRewrite 的强化学习系统采用模块化设计，主要包含以下核心组件：

```
输入: 粗笔画序列 (.npy) + 书法图像 (.png)
    ↓
┌─────────────────────────────────┐
│  CalliEnv 环境                   │
│  • 加载图像和粗笔画               │
│  • 工具物理建模                   │
│  • 状态计算                       │
└─────────────────────────────────┘
    ↓ 状态 (8维)
┌─────────────────────────────────┐
│  SAC 策略网络                     │
│  • Actor: 状态 → 动作             │
│  • Critic: 状态-动作 → Q值        │
└─────────────────────────────────┘
    ↓ 动作 (2维)
┌─────────────────────────────────┐
│  环境交互 & 渲染                  │
│  • 执行动作                       │
│  • 计算新笔画参数                 │
│  • 渲染到画布                     │
└─────────────────────────────────┘
    ↓ 奖励 (标量)
┌─────────────────────────────────┐
│  奖励计算                         │
│  • 图像相似度                     │
│  • 笔画平滑度                     │
│  • 笔画大小控制                   │
└─────────────────────────────────┘
    ↓ 策略更新
┌─────────────────────────────────┐
│  SAC 优化                        │
│  • Actor 损失                    │
│  • Critic 损失                   │
│  • 温度参数自适应                │
└─────────────────────────────────┘
    ↓ 多轮迭代
输出: 优化后笔画序列 (.npy)
```

## 🎯 核心组件

### 1. CalliEnv - 自定义书法环境

**文件位置**: `rl_finetune/Callienv/envs/Callienv.py`

CalliEnv 是一个符合 OpenAI Gym 接口标准的强化学习环境，专为书法任务设计。

#### 状态空间 (8维)
```python
state = [period, r, l, theta, curvature, r_prime, vec_x, vec_y]
```

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `period` | [0, 1] | 当前笔画进度 (current_pos / stroke_length) |
| 1 | `r` | [0, 1] | 笔刷半径/宽度 (归一化) |
| 2 | `l` | [0, 1] | 笔刷长度 (归一化) |
| 3 | `theta` | [0, 1] | 旋转角度 (归一化到 [0, 360°]) |
| 4 | `curvature` | [0, 1] | 当前曲率 (通过 sin 计算) |
| 5 | `r_prime` | [-1, 1] | 上一步移动距离 |
| 6 | `vec_x` | [-1, 1] | 未来方向单位向量 X |
| 7 | `vec_y` | [-1, 1] | 未来方向单位向量 Y |

#### 动作空间 (2维)
```python
action = [r_prime, theta_prime]
```

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `r_prime` | [-1, 1] | 移动距离 (× r_prime_bound = 0.022) |
| 1 | `theta_prime` | [-1, 1] | 移动角度 (× π) |

#### 工具物理建模

CalliEnv 支持多种书写工具的物理建模：

1. **毛笔 (Writing_Brush)**: 圆形笔头 + 三角形笔尖，真实物理动力学
2. **椭圆笔刷 (Ellipse)**: 固定椭圆形状，可旋转
3. **凿尖马克笔 (Chisel_Tip_Marker)**: 矩形笔头，固定尺寸

### 2. 策略网络

**文件位置**: `rl_finetune/MLP/model.py`

使用 My_MLP 网络架构，支持多种变体：

- **标准 MLP**: 2-3 层隐藏层，ReLU 激活
- **Fourier Features**: 支持傅里叶特征映射，增强高频信息捕捉
- **SIREN**: 周期性激活函数网络，适合处理周期性数据

**网络结构** (在 try_tianshou.py 中配置):
- Actor网络: [128, 256] 隐藏层
- Critic网络: [128, 256, 256] 隐藏层
- 优化器: Adam，学习率 actor=3e-5, critic=1e-4

### 3. 训练系统

**文件位置**: `rl_finetune/try_tianshou.py`

使用 Tianshou（天授）深度强化学习库实现训练流程：

#### 核心算法：SAC (Soft Actor-Critic)

SAC 是一种高效的 off-policy 强化学习算法，具有以下优势：

- **高样本效率**: 离线策略，重复利用经验
- **稳定性**: 熵正则化 + 双 Q 网络
- **探索能力**: 自动调节温度参数
- **适用于连续动作**: 无需离散化

#### 训练阶段

1. **环境初始化**: 创建并行训练和测试环境
2. **数据预填充**: 使用随机策略预填充经验回放池
3. **SAC 训练**: 执行 off-policy 训练，每收集一定步数更新网络
4. **模型保存**: 训练完成后保存策略网络
5. **测试评估**: 在测试集上验证训练效果

## 📊 训练配置

### 超参数设置

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_epoch` | 150 | 最大训练轮数 |
| `step_per_epoch` | 10000 | 每轮最多收集步数 |
| `step_per_collect` | 16 | 收集多少步后更新网络 |
| `update_per_step` | 2.0 | 每步更新次数 |
| `batch_size` | 2048 | 批次大小 |
| `buffer_size` | 2^20 | 经验回放池大小 |
| `actor_lr` | 3e-5 | Actor 学习率 |
| `critic_lr` | 1e-4 | Critic 学习率 |
| `tau` | 0.005 | 目标网络软更新系数 |
| `gamma` | 0.9 | 折扣因子 |

### 工具属性配置

**文件位置**: `rl_finetune/tool_property/`

#### 毛笔配置 (brush.json)
```json
{
    "r_min": 0,
    "r_max": 0.064,
    "l_min": 0,
    "l_max": 0.17875,
    "theta_min": 0,
    "theta_max": 360,
    "theta_step": 10
}
```

## 🚀 训练命令

### 1. 使用脚本训练

```bash
cd rl_finetune

# 毛笔训练
bash scripts/train_brush.sh

# 椭圆笔刷训练
bash scripts/train_ellipse.sh

# 凿尖马克笔训练
bash scripts/train_marker.sh
```

### 2. 直接运行训练脚本

```bash
cd rl_finetune

python try_tianshou.py --train_data_dir ./data/train_data/ --test_data_dir ./data/test_data/ --which_tool brush --tool_property_dir ./tool_property/brush.json --logdir ./result/output.log
```

### 3. 完整参数训练

```bash
python try_tianshou.py \
    --train_data_dir ./data/train_data/ \
    --test_data_dir ./data/test_data/ \
    --which_tool brush \
    --tool_property_dir ./tool_property/brush.json \
    --logdir ./result/output.log \
    --max_epoch 150 \
    --step_per_epoch 10000 \
    --step_per_collect 16 \
    --batch_size 2048 \
    --update_per_step 2.0 \
    --lr_actor 3e-5 \
    --lr_critic 1e-4
```

## 📈 监控训练

### TensorBoard 可视化

```bash
cd rl_finetune
tensorboard --logdir ./result/output.log --port 6006
```

访问 http://localhost:6006 查看训练过程中的指标：

- **奖励曲线**: 训练和测试的平均奖励
- **损失曲线**: Actor 和 Critic 的学习损失
- **图像记录**: 目标图像和渲染图像的对比
- **视频录制**: 训练过程的视频（在 ./result/demo/videos/ 目录）

### 训练输出文件

训练完成后，结果保存在以下位置：

```
rl_finetune/
├── result/
│   ├── output.log/              # TensorBoard 日志
│   └── demo/
│       ├── models/              # 策略网络模型
│       ├── videos/              # 训练视频
│       └── arrays/              # 优化后的笔画序列 (.npy)
└── data/
    ├── train_data/              # 训练数据
    └── test_data/               # 测试数据
```

## 🎯 训练策略详解

### 1. 渐进式优化

使用 EMA（指数移动平均）技术平滑优化轨迹，避免剧烈变化：

```python
# 在 episode 结束时更新粗笔画
if terminated:
    if diff >= start_update and diff % update == 0:
        self.skel_list = EMA(self.skel_list,
                             self.new_skel_list.reshape(-1, 2),
                             self.ema_gamma)
```

**公式**: `skel_new = γ × skel_old + (1-γ) × skel_predicted`

### 2. 多图像循环训练

每张图像重复训练多次（image_iter=10）：

```python
if self.counter % self.image_iter == 0:
    # 切换到新图像
    count = (self.counter // self.image_iter) % self.list_num
    pick_data = self.data_pool[count]
    self.load_skel_and_img(pick_data, count)
```

### 3. 并行训练

使用多进程环境加速训练：

```python
from tianshou.env import SubprocVectorEnv

# 创建 4 个并行训练环境
train_envs = SubprocVectorEnv([
    lambda i=i: RecordVideo(
        gym.make('CalliEnv-v0',
                 tool=tool,
                 folder_path=args.train_data_dir,
                 env_rank=(i*100, (i+1)*100),
                 render_mode='rgb_array'),
        video_folder=save_video_dir,
        name_prefix=f'trainvids_{i}'
    ) for i in range(4)
])
```

## 🔧 常见问题

### Q: 为什么使用 SAC 而不是其他算法？

**A**: SAC 的优势：
- **高样本效率**: 离线策略，重复利用经验
- **稳定性**: 熵正则化 + 双 Q 网络
- **探索能力**: 自动调节温度参数
- **适用于连续动作**: 无需离散化

### Q: 如何选择合适的 image_iter 值？

**A**: 经验法则：
- 简单字符（笔画少）: 15-20
- 复杂字符（笔画多）: 20-30
- 数据集小: 增大 iter（充分利用）
- 数据集大: 减小 iter（避免过拟合）

### Q: 训练不收敛怎么办？

**A**: 检查列表：
1. **奖励设计**: 确保终止奖励占主导
2. **学习率**: 尝试降低到 1e-4
3. **batch_size**: 尝试增大到 256
4. **EMA 参数**: 增大 gamma 到 0.98
5. **环境数**: 增加并行环境到 12-16

### Q: 如何在训练过程中保存最佳模型？

**A**: Tianshou 训练器会自动保存最佳模型。可以在训练脚本中设置：

```python
result = offpolicy_trainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    save_best_fn=lambda (epoch, reward, policy): reward > best_reward,
    save_checkpoint_fn=lambda (epoch, reward, policy): epoch % 10 == 0
)
```

## 🔄 与 seq_extract 的对比

| 特性 | seq_extract | rl_finetune |
|------|-------------|-------------|
| **方法** | 监督学习 (LSTM) | 强化学习 (SAC) |
| **输入** | 书法图像 | 粗笔画 + 图像 |
| **输出** | 粗笔画序列 | 精细笔画序列 |
| **训练数据** | QuickDraw + 书法 | 书法数据 |
| **训练时间** | ~5天 | ~6小时 |
| **物理建模** | 无 | 工具动力学 |
| **适应性** | 通用 | 工具特定 |

## 📚 进一步阅读

### 详细文档

1. **Tianshou 和 Gym 使用指南**: `docs/Tianshou_Gym_Guide.md`
2. **rl_finetune 训练过程详解**: `rl_finetune/TRAINING_PROCESS.md`
3. **环境修改说明**: `rl_finetune/modify_env.md`
4. **序列提取训练过程**: `seq_extract/TRAINING_PROCESS.md`

### 核心代码文件

- **训练主程序**: `rl_finetune/try_tianshou.py`
- **环境实现**: `rl_finetune/Callienv/envs/Callienv.py`
- **工具建模**: `rl_finetune/Callienv/envs/tools.py`
- **网络架构**: `rl_finetune/MLP/model.py`
- **辅助工具**: `rl_finetune/utils.py`
- **骨骼处理**: `rl_finetune/Callienv/envs/skel_utils.py`

---

一阶段输出 .npz (4.npz)

形状: (60, 7)
格式: [pen_state, x1, y1, x2, y2, radius, scaling]

示例数据:
  [0] 笔状态=1 (移动), (0.340,0.454)->(0.203,0.830), r=0.307
  [1] 笔状态=1 (移动), (0.054,0.781)->(0.122,0.359), r=0.217
  [2] 笔状态=0 (绘画), (0.396,0.177)->(0.936,-0.073), r=0.199

统计: 30个绘画stroke, 30个移动stroke
二阶段使用的 .npy (5.npy)

形状: (53, 7)
格式: [pen_state, x, y, r, col1, col2, col3]

示例数据:
  [0] 笔状态=1 (新笔画), (x=45.771, y=95.578), r=54.273
  [1] 笔状态=0 (绘画), (x=45.771, y=95.578), r=54.273
  [2] 笔状态=0 (绘画), (x=84.611, y=82.264), r=86.404

注意: 这里的 x,y,r 是像素坐标 (0-256)，不是归一化的！
🔄 数据流程总结

seq_extract 阶段:
  输入: 书法图像
  输出: npz 文件，包含相对坐标的贝塞尔曲线笔画
         [pen_state, x1_rel, y1_rel, x2_rel, y2_rel, r, s]

rl_finetune 阶段:
  输入: npz + 图像
  处理: make_global_nplist() 把相对坐标转绝对像素坐标
        parse_skel() 采样贝塞尔曲线为密集点
  输出: npy 文件
         [pen_state, x_px, y_px, r_px, col1, col2, col3]

一阶段 npz: strokes_data (60,7)                                      │
│ [pen_state, x1_rel, y1_rel, x2_rel, y2_rel, r, s]                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼ make_global_nplist()
┌─────────────────────────────────────────────────────────────────────┐
│ 中间格式 (60,7)                                                      │
│ [pen_state, x0_px, y0_px, x1_px, y1_px, x2_px, y2_px]               │
│   (绝对像素坐标 0-256)                                               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼ parse_skel() - 采样贝塞尔曲线
┌─────────────────────────────────────────────────────────────────────┐
│ 密集点格式 (150,3)                                                  │
│ [pen_state, x_norm, y_norm]                                          │
│   (归一化 0-1)                                                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼ add_beg_end_seq() - 可选延伸
┌─────────────────────────────────────────────────────────────────────┐
│ 扩展点格式 (180,3)                                                  │
│ [pen_state, x_norm, y_norm]                                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼ CalliEnv 内部处理
┌─────────────────────────────────────────────────────────────────────┐
│ 二阶段运行时                                                         │
│ self.skel_list: (180,2) → 坐标 [x, y]                               │
│ self.pt_list: (180,) → pen_state                                     │
│ r: 在 step() 中动态计算 (通过 tool.geometric_r_l())   


**文档版本**: v1.0
**最后更新**: 2026-05-18
**作者**: Claude Code

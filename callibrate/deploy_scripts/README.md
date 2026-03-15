# 真机部署验证脚本

CalliRewrite Franka 机器人真机迁移的分阶段验证流程。

**核心原则**：从完全离线到实际接触，逐步提升风险等级，每步通过后才进入下一步。

---

## 快速参考

| 步骤 | 脚本 | 需要机器人 | 风险 |
|------|------|-----------|------|
| 1 | `01_check_dependencies.py` | 否 | 无 |
| 2 | `02_check_npz.py` | 否 | 无 |
| 3 | `03_check_config.py` | 否 | 无 |
| 4 | `04_simulate_mujoco.py` | 否 | 无 |
| 5 | `05_test_connection.py` | 是（通电不解锁也可） | 极低（无运动） |
| 6 | `06_teach_workspace.py` | 是（手动导引） | 低（人工导引） |
| 7 | `07_first_move.py` | 是（自动运动） | 中（首次运动） |
| 8 | `08_single_point_test.py` | 是（接触纸面） | 中（轻触） |
| 9 | `09_calibration_run.py` | 是（完整轨迹） | 中高（多点） |

---

## 阶段 0 — 离线验证（无需机器人）

### Step 1 · 依赖检查

**作用**：检查所有 Python 包、Franka 库、配置文件、NPZ 示例文件是否就绪。

**执行命令**：
```bash
cd /Users/seer/CalliRewrite/callibrate/deploy_scripts
python 01_check_dependencies.py
```

**通过标准**：输出最后一行显示 `✅ 依赖检查通过`，无 ❌。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ numpy / yaml / matplotlib 未安装` | 环境缺包 | `pip install numpy pyyaml matplotlib scipy opencv-python scikit-image` |
| `❌ Python 3.x — 需要 3.10+` | Python 版本太低 | `conda create -n calli python=3.11 && conda activate calli` |
| `⚠️ franky/frankx 未安装` | Franka 控制库缺失 | 阶段 0 不影响；上机前运行 `pip install frankx` |
| `⚠️ mujoco 未安装` | 仿真库缺失 | Step 4 需要：`pip install mujoco` |
| `❌ franka_config.yaml 不存在` | 配置文件缺失 | `cp callibrate/franka_config.yaml.example callibrate/franka_config.yaml`，或手动创建（见下方模板） |
| `❌ franka_config.yaml 中 workspace_center 字段不完整` | 配置未填写 | 编辑 `callibrate/franka_config.yaml`，填入 `x/y/z` 三个字段 |
| `⚠️ workspace_center.x 超出典型范围` | 坐标可能填错 | 典型值：x ∈ [0.3, 0.7]，y ∈ [-0.3, 0.3]，z ∈ [0.1, 0.5]，用 Step 6 实测 |
| `⚠️ examples/xxx.npz 不存在` | 示例文件缺失 | 用 Step 4 的 MuJoCo 脚本生成，或从 `seq_extract/outputs/` 复制一个 |

**franka_config.yaml 最小模板**：
```yaml
robot_ip: 172.16.0.2
workspace_center:
  x: 0.4      # 实测后填写，这是占位默认值
  y: 0.0
  z: 0.2
default_speed: 0.05
z_offset: -0.09
```

---

### Step 2 · NPZ 轨迹文件检查

**作用**：验证将要上机执行的 NPZ 文件坐标范围合理、Z 轴分布正常、无高风险异常点。

**执行命令**：
```bash
# 检查 examples/ 目录下所有 NPZ
python 02_check_npz.py

# 检查指定文件
python 02_check_npz.py ../examples/example_永.npz

# 检查并弹出可视化图
python 02_check_npz.py ../examples/example_永.npz --plot
```

**通过标准**：
- X/Y 范围 < 15 cm（相对坐标，字符大小正常）
- Z 范围 -0.01 ~ 0.05 m（v2 格式），或 -0.10 ~ 0.05 m（calibrate.py 格式）
- 无 `❌ 高风险` 警告

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ 文件不存在` | 路径错误 | 确认文件路径，或先运行 Step 4 生成 NPZ |
| `❌ Z 轴最小值 < -0.15 m` | Z 值异常偏低（会戳穿纸） | 检查 NPZ 是否由 `calibrate.py` 生成（含 -0.09 偏移），确认使用的是哪种格式 |
| `⚠️ 单步 XY 跳变 > 20 cm` | 轨迹中有非预期大跳变 | 打开 `--plot` 查看俯视图，定位异常点索引，检查生成轨迹的参数 |
| 无抬笔点（全部 Z 为负） | NPZ 缺少 pen-up 动作 | 重新生成 NPZ，确认 `convert_rl_to_npz()` 的 style_type 参数正确 |
| 抬笔点 Z 过低（< 0.02 m） | 抬笔不足，路径规划时会擦纸 | 检查 NPZ 生成逻辑中的抬笔高度参数 |

---

### Step 3 · 配置文件检查与坐标预览

**作用**：验证 `franka_config.yaml` 有效，并预览 NPZ 坐标被映射到机器人后的实际物理位置。

**执行命令**：
```bash
python 03_check_config.py

# 指定 NPZ 文件（查看该文件映射后的范围）
python 03_check_config.py --npz ../examples/example_永.npz
```

**通过标准**：
- workspace_center 在 Franka 合法范围内
- 预览的机器人坐标 X ∈ [0.2, 0.8]，Y ∈ [-0.5, 0.5]，Z ∈ [0.05, 0.5]

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `workspace_center.x = 0.4`（默认值未修改） | Step 6 尚未执行 | 先做 Step 6 示教，完成后回来重跑 Step 3 确认 |
| 预览 Z 范围偏高（机器人到不了纸面） | workspace_center.z 填写偏高 | 减小 z 值（如从 0.3 改为 0.2），重跑 Step 6 实测 |
| 预览 Z 范围极低（负值很大） | NPZ 包含 calibrate.py 的 -0.09 偏移，且 workspace_center.z 未对齐 | 确认 NPZ 类型：v2 格式不含偏移，直接加 workspace_center.z；calibrate 格式已含偏移，workspace_center.z 设为 0 |
| 预览机器人 X 超出 Franka 臂长 | workspace_center 位置过远 | Franka Panda 最大臂展约 0.85m，workspace_center.x 建议 ≤ 0.65 |

---

### Step 4 · MuJoCo 离线仿真

**作用**：在不启动机器人的情况下，仿真轨迹并保存画布图像，肉眼确认字形正确。

**执行命令**：
```bash
# 仿真默认示例（examples/example_永.npz）
python 04_simulate_mujoco.py

# 仿真指定文件
python 04_simulate_mujoco.py path/to/your.npz

# 仿真 seq_extract 输出目录下所有 v2 NPZ
python 04_simulate_mujoco.py --all

# 调整仿真速度（越大越快，不影响轨迹）
python 04_simulate_mujoco.py --speed 0.1
```

**输出**：`deploy_scripts/sim_outputs/<filename>_canvas.png`

**通过标准**：用图片查看器打开画布，字形轮廓清晰、笔画方向正确。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ 导入失败: No module named 'mujoco'` | MuJoCo 未安装 | `pip install mujoco` |
| `❌ 导入失败: No module named 'cv2'` | OpenCV 未安装 | `pip install opencv-python` |
| `❌ 导入失败: No module named 'mujoco_simulator'` | mujoco_sim 路径问题 | 脚本中 `SIM_DIR` 指向 `../../mujoco_sim`，确认目录存在：`ls /Users/seer/CalliRewrite/mujoco_sim/` |
| 画布图像全白（无笔迹） | Z 值偏高，笔尖未接触仿真纸面 | 检查 NPZ 的 Z 范围（应包含负值或接近 0 的值），降低 `z_world = z_pts[i] + 0.011` 中的偏移量 |
| 字形镜像或旋转 | 坐标轴方向与预期不符 | 检查 `mujoco_simulator.py` 中的 `paper_offset` 和坐标系方向 |
| 仿真极慢 | `--speed` 默认 0.05，控制点多时耗时长 | 改为 `--speed 0.2` 加速仿真（不影响真机执行） |

---

## 阶段 1 — 连接测试（机器人通电，不运动）

### Step 5 · 网络与 Franka 库连接测试

**前置条件**：
- 机器人通电
- 电脑与机器人在同一子网（通常用网线直连或同一交换机）
- （可选）Franka Desk 中机器人已解锁（显示 Ready）

**执行命令**：
```bash
# 使用配置文件中的 IP
python 05_test_connection.py

# 指定 IP
python 05_test_connection.py --ip 172.16.0.2
```

**通过标准**：Ping 成功 + `✅ franky/frankx 连接成功`。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ 无法 ping 通` | 网络不通 | 1. 检查网线是否插好；2. 确认 IP 正确（Franka Desk → Settings → Network）；3. 临时关闭电脑防火墙测试 |
| Ping 通但 `❌ franky/frankx 连接失败` | 机器人未解锁 | 打开 Franka Desk（`https://<ip>`），点击解锁按钮，确认状态变为 Ready |
| `Franka Desk 仍然打开且占用连接` | 浏览器占用了控制权 | 关闭所有 Franka Desk 浏览器标签页，再重试 |
| `❌ 没有 franky 也没有 frankx` | Franka 库未安装 | `pip install frankx`，或参考 franky 官方文档编译安装 |
| 连接超时 | 网络延迟高 | 改用网线直连；检查交换机配置 |
| `Robot is not ready` 报错 | 机器人处于错误状态 | Franka Desk 中查看错误信息，点击「清除错误」，必要时重新上电 |

---

## 阶段 2 — 工作台示教（手动导引，不自动运动）

### Step 6 · 工作台坐标教点

**作用**：引导手动把笔尖导引到纸面中心，读取此时末端坐标，写入 `franka_config.yaml`。

**前置条件**：Step 5 通过，Franka Desk 可访问。

**执行命令**：
```bash
# 标准模式（读取坐标并写入配置）
python 06_teach_workspace.py

# 指定 IP
python 06_teach_workspace.py --ip 172.16.0.2

# 仅读取坐标，不写入配置（测试用）
python 06_teach_workspace.py --dry-run
```

**操作步骤**：
1. 打开 Franka Desk（`https://172.16.0.2`）
2. 解锁机器人 → 切换到「引导模式」（按住末端黑色导引按钮）
3. 用手把笔尖导引到纸张中央，笔尖轻碰纸面
4. 保持末端静止，在终端按 Enter
5. 脚本读取坐标并询问是否写入，输入 `y` 确认

**通过标准**：`franka_config.yaml` 中 `workspace_center` 更新为实测坐标。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ 读取机器人位置失败` | 机器人未处于引导模式或未连接 | 确认 Franka Desk 中切换到了引导模式（末端导引按钮高亮），重试 |
| `franky 读取失败: attribute 'O_T_EE'` | franky API 版本差异 | 改用 frankx：`pip install frankx`，脚本会自动 fallback |
| `frankx: column-major 读取错位` | frankx O_T_EE 格式为列主序 | 脚本已处理（`t[12], t[13], t[14]`），如坐标异常可加 `--dry-run` 手动核对 |
| 读取的 Z 值和预期纸面高度差异大 | 笔尖末端坐标包含工具偏移 | 若使用了工具坐标系，需减去工具长度偏移；最简单方案：笔尖接触纸面时直接读取 Z |
| 无法切换引导模式 | 机器人未解锁或处于错误状态 | Franka Desk 中先解锁，清除所有错误再切换 |
| **备用方案**：脚本失败时 | 任意原因 | 在 Franka Desk 「当前位置」页面手动记录 x/y/z，直接编辑 `franka_config.yaml` |

---

## 阶段 3 — 首次自动运动（不接触纸面）

### Step 7 · 首次运动：仅移到纸面上方 10 cm

**作用**：验证机器人可以按配置坐标自动运动，目标位置在纸面正上方 10 cm，不接触纸面。

**前置条件**：Step 6 完成，workspace_center 已实测填写。

**安全要求**：
- **手放在急停按钮上**，全程准备按下
- 机器人周围 50 cm 内无人无障碍物
- Franka Desk 中机器人状态为 Ready

**执行命令**：
```bash
python 07_first_move.py
# 输入 YES 确认后机器人开始运动（speed=0.02，极慢）
```

**通过标准**：机器人缓慢运动到纸面正上方约 10 cm，目视位置正确，无报错。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| `❌ 连接失败` | Franka 库连接问题 | 重跑 Step 5，排查连接问题 |
| `❌ Motion failed` | 目标位置超出机器人工作空间 | 检查 workspace_center 是否在 Franka 合法范围内（Step 3 预览），调整 x/y/z |
| 机器人运动方向与预期不符 | workspace_center x/y/z 填写顺序错误 | 对照 Franka 坐标系（x 指向正前方，z 竖直向上），重新示教 Step 6 |
| 机器人运动到的位置偏高/偏低 | workspace_center.z 与实际纸面不符 | 重新执行 Step 6，更准确地示教纸面坐标 |
| 机器人运动中途停止并报 `Franka error` | 碰撞检测触发 | 立即检查是否有障碍物，清除错误后从 Franka 安全位置重试 |
| 机器人抖动或异常 | 速度太快或动力学参数问题 | 当前 speed=0.02（最大速度的 2%），应该非常缓慢；若仍抖动，检查 Franka Desk 中的控制模式 |

---

## 阶段 4 — 单点接触测试（轻触纸面）

### Step 8 · 轻触纸面中心一次

**作用**：验证 workspace_center.z 与实际纸面一致，笔尖接触力合理。

**执行路径**：`Home → 纸面上方5cm → 纸面-1mm（轻触）→ 纸面上方5cm → Home`

**前置条件**：Step 7 通过，纸已放在工作台上。

**执行命令**：
```bash
python 08_single_point_test.py
# 输入 YES 开始，下降到纸面时脚本会询问接触情况
```

**通过标准**：纸面出现轻微印记，接触力适中（不会戳穿纸，不会悬空）。

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| 笔尖没碰到纸（悬空） | workspace_center.z 偏高 | 减小 `franka_config.yaml` 中 `workspace_center.z`（如减 0.005），重试 |
| 笔尖压力太大（戳穿纸/机器人报错） | workspace_center.z 偏低 | 增大 `workspace_center.z`（如加 0.003），重试；**如有报错先按急停** |
| 接触偏移（不在纸中央） | workspace_center.x/y 不准 | 重新执行 Step 6，更精确地示教 x/y 位置 |
| `❌ Motion failed` 下降时报错 | 碰撞检测阈值触发（笔尖接触纸面产生的力） | 适当增大 workspace_center.z（减少压深），或在 Franka Desk 调高碰撞阈值 |
| 机器人下降太快 | speed 参数 | 当前已设为 0.01（极慢）；如仍快，检查 franky/frankx 的速度单位是否理解正确 |

**Z 轴调整参考**：
```
观察结果       调整方向         调整量
笔尖悬空 1cm → 减小 z        → -0.010
笔尖悬空 3mm → 减小 z        → -0.003
接触正常      → 不调整
压力稍大      → 增大 z        → +0.002
压力很大/报错  → 增大 z        → +0.005 或更多
```

---

## 阶段 5 — 标定线执行（完整坐标映射验证）

### Step 9 · 在纸上画 5 条横线

**作用**：执行完整坐标映射，验证 X/Y 方向和 Z 压力梯度均正确。5 条横线使用不同 Z 深度（轻到重）。

**前置条件**：Step 8 通过，workspace_center.z 已校准。

**执行命令**：
```bash
python 09_calibration_run.py
# 会显示坐标映射预览，输入 YES 开始执行
```

**输出**：`deploy_scripts/calib_lines.npz`（标定轨迹文件）

**通过标准**：
- [ ] 5 条线沿 X 轴方向（水平方向）
- [ ] 线间距约 8 mm，排列均匀
- [ ] 从第 1 条到第 5 条，线从细到粗（Z 压力递增）
- [ ] 线迹连续，无断墨

**可能的问题与补救**：

| 问题 | 原因 | 补救 |
|------|------|------|
| 线方向是竖的（沿 Y 轴） | 机器人坐标系 X/Y 与预期相反 | 交换 `franka_config.yaml` 中 x/y 值，重新示教；或修改轨迹生成中 x/y 的含义 |
| 5 条线全部重叠 | Y 方向映射问题 | 检查 `map_to_robot()` 中 y_npz 的范围，确认 y_offset（0~32mm）被正确映射 |
| 线全部一样粗（无粗细变化） | Z 压力差异太小 | 检查实际接触时机器人 Z 变化量，减小 workspace_center.z 以增大压深范围 |
| 每条线都只有起点印记（无连续线） | 运动速度太慢或连接中断 | 检查中间是否有运动失败打印；speed 可从 0.02 适当提高到 0.03 |
| 线偏离纸中央 | workspace_center.x/y 示教不准 | 重新执行 Step 6 精确示教 |
| 执行中 `❌ 第 N 个点运动失败` | 坐标超出工作空间或碰撞报错 | 检查映射后的机器人坐标范围（脚本启动时会打印），确认在合法范围内 |

---

## Step 9 通过后 — 完整字符执行

所有 9 步通过后，可以执行实际的书法字符 NPZ：

```bash
# 方法 1：直接用 RoboControl.py
python ../RoboControl.py <your_npz_path> 172.16.0.2 0.03

# 方法 2：通过 LingBot-VLA 自然语言驱动
cd /Users/seer/CalliRewrite
python -m lingbot_planner.pipeline "写一个永字" --execute --robot-ip 172.16.0.2
```

> ⚠️ **注意 RoboControl.py 已知问题**：直接使用 `Control()` 时，workspace_center 会从 NPZ 坐标自动计算（约 0.06m），与实际机器人坐标（约 0.4m）不符。建议通过 `FrankaCalligraphyController` 类手动传入 workspace_center，参考 Step 9 脚本中的 `map_to_robot()` 函数。

---

## 坐标映射说明

NPZ 文件中的坐标是**相对坐标**（以字符中心为原点），各脚本统一使用以下公式映射到机器人绝对坐标：

```python
x_c = (x_npz.min() + x_npz.max()) / 2
y_c = (y_npz.min() + y_npz.max()) / 2

x_robot = workspace_center_x + (x_npz - x_c)   # 字符中心对齐到 workspace_center
y_robot = workspace_center_y + (y_npz - y_c)
z_robot = workspace_center_z + z_npz            # z_npz=0 → 纸面；负值 → 压笔；正值 → 抬笔
```

`workspace_center` 从 `callibrate/franka_config.yaml` 读取，由 Step 6 实测填写。

---

## 输出文件汇总

| 路径 | 由哪步生成 | 内容 |
|------|-----------|------|
| `deploy_scripts/sim_outputs/*.png` | Step 4 | MuJoCo 仿真画布图像 |
| `deploy_scripts/calib_lines.npz` | Step 9 | 5 条标定横线轨迹 |
| `callibrate/franka_config.yaml` | Step 6（写入） | 工作台坐标（workspace_center） |

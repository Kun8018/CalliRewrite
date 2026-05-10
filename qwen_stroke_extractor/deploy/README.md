# qwen_stroke_extractor 服务器部署指南

本目录包含一套完整的服务器部署脚本，用于在服务器上快速部署 qwen_stroke_extractor。

## 🚀 快速开始

### 方法一：一键部署（推荐）

```bash
cd /path/to/CalliRewrite/qwen_stroke_extractor/deploy

# 赋予执行权限
chmod +x *.sh

# 一键部署
bash deploy_all.sh
```

### 方法二：分步部署

```bash
# 1. 安装依赖
bash 01_install_dependencies.sh

# 2. 下载模型
bash 02_download_model.sh Qwen-VL

# 3. 测试本地模型
bash 03_test_local_model.sh

# 4. 批量处理
bash 04_batch_process.sh /path/to/images

# 5. 查看指南
bash 05_deploy_guide.sh
```

## 📋 脚本说明

| 脚本 | 说明 |
|------|------|
| `deploy_all.sh` | 一键部署，自动化执行所有步骤 |
| `01_install_dependencies.sh` | 安装 Python 依赖 |
| `02_download_model.sh` | 从 HuggingFace 下载模型 |
| `03_test_local_model.sh` | 测试本地模型是否正常工作 |
| `04_batch_process.sh` | 批量处理书法图像 |
| `05_deploy_guide.sh` | 显示部署指南和检查清单 |

## 🔧 硬件要求

### 最低配置

- **CPU**: 4 核以上
- **内存**: 16GB
- **磁盘**: 50GB 可用空间
- **GPU**: 无要求（使用 API 模式）

### 推荐配置

- **CPU**: 8 核以上
- **内存**: 32GB
- **磁盘**: 100GB 可用空间
- **GPU**: NVIDIA 显卡，显存 ≥16GB
- **CUDA**: 11.8+

## 💡 模型选择

| 模型 | 大小 | 显存要求 | 推荐用途 |
|------|------|----------|----------|
| Qwen-VL | ~10GB | 10-16GB | 推荐，平衡性能与资源 |
| Qwen-VL-Plus | ~30GB | 20-32GB | 最好效果，需要更多资源 |

### 下载不同模型

```bash
# 下载 Qwen-VL（推荐）
bash 02_download_model.sh Qwen-VL

# 或下载 Qwen-VL-Plus
bash 02_download_model.sh Qwen-VL-Plus
```

## 🌐 API 模式（如果 GPU 不足）

如果服务器没有足够的 GPU，推荐使用 API 模式：

```python
# 使用 API 模式
from qwen_stroke_extractor.extractor import create_extractor

extractor = create_extractor(
    use_api=True,
    api_key="your-api-key"
)

result = extractor.extract("calligraphy.png")
```

获取 API Key：
1. 访问 https://dashscope.aliyun.com/
2. 注册阿里云账号
3. 创建 API Key

## 📊 检查清单

部署前确认：

- [ ] 服务器有 NVIDIA GPU（可选）
- [ ] CUDA 已正确安装（可选）
- [ ] 有足够的磁盘空间（> 50GB）
- [ ] 项目已正确克隆到服务器
- [ ] 有虚拟环境或 conda 环境

部署后检查：

- [ ] 依赖安装成功
- [ ] 模型下载完整
- [ ] 测试图像处理正常
- [ ] 批量处理能正常工作

## 🎯 使用示例

### 1. 处理单张图像

```bash
cd /path/to/CalliRewrite

# 激活虚拟环境
source calli_train_env/bin/activate

# 使用本地模型处理
cd qwen_stroke_extractor
python examples.py /path/to/image.png --model-path models/Qwen-VL --visualize
```

### 2. 批量处理

```bash
cd /path/to/CalliRewrite/qwen_stroke_extractor/deploy
bash 04_batch_process.sh /path/to/images
```

### 3. 在代码中使用

```python
from qwen_stroke_extractor.extractor import create_extractor

# 初始化
extractor = create_extractor(
    model_path="./models/Qwen-VL",
    use_api=False
)

# 提取笔画
result = extractor.extract("calligraphy.png")

# 保存结果
extractor.save_result(result, "strokes.npy", format="npy")

# 可视化
extractor.visualize_result(
    result,
    "viz.png",
    background_image="calligraphy.png"
)
```

## 🔍 常见问题

### 问题 1: 模型下载太慢

**解决方法**：
- 使用 ModelScope（国内镜像）
- 使用阿里云 OSS 加速器
- 使用 VPN

### 问题 2: 显存不足（OOM）

**解决方法**：
- 使用更小的模型（Qwen-VL）
- 使用 CPU 模式（虽慢但可用）
- 切换到 API 模式

### 问题 3: 依赖安装失败

**解决方法**：
```bash
# 更新 pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 4: git-lfs 未安装

**解决方法**：
```bash
# Ubuntu/Debian
apt-get install -y git-lfs

# CentOS/RHEL
yum install -y git-lfs

# macOS
brew install git-lfs

# 然后初始化
git lfs install
```

## 📁 目录结构

```
CalliRewrite/
├── qwen_stroke_extractor/
│   ├── deploy/                  # 部署脚本目录
│   │   ├── 01_install_dependencies.sh
│   │   ├── 02_download_model.sh
│   │   ├── 03_test_local_model.sh
│   │   ├── 04_batch_process.sh
│   │   ├── 05_deploy_guide.sh
│   │   ├── deploy_all.sh
│   │   └── README.md
│   ├── models/                  # 模型存储目录（自动创建）
│   │   ├── Qwen-VL/
│   │   └── Qwen-VL-Plus/
│   ├── extractor.py
│   └── examples.py
├── calli_train_env/            # 虚拟环境
└── outputs/
    └── qwen_results/           # 输出目录（自动创建）
```

## 📚 更多文档

- [qwen_stroke_extractor README](../README.md)
- [项目主 README](../../README.md)
- [千问模型文档](https://github.com/QwenLM/Qwen-VL)
- [阿里云 DashScope](https://dashscope.aliyun.com/)

## 🤝 获取帮助

如果遇到问题：

1. 先运行 `bash 05_deploy_guide.sh` 检查环境
2. 查看 [常见问题](#-常见问题) 部分
3. 检查 GitHub Issues
4. 查看 Qwen 官方文档

## 📄 许可证

与 CalliRewrite 项目保持一致。

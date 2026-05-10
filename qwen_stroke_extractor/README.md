# Qwen 千问Plus 笔画提取器

基于千问Plus（Qwen-VL）多模态大模型的书法图像到参数化笔画序列提取工具。

## 功能特点

- **图像识别**: 使用千问多模态模型分析书法图像
- **笔画提取**: 自动识别笔画并按书写顺序排列
- **参数化输出**: 输出归一化坐标、宽度等参数
- **格式兼容**: 支持与原 CalliRewrite 项目兼容的输出格式

## 两种使用方式

### 1. API 模式（推荐）

使用阿里云 DashScope API，无需本地部署大模型。

**优点**:
- 不需要下载大模型（约几十GB）
- 不需要高性能 GPU
- 即时可用

**缺点**:
- 需要阿里云账号
- 需要 API 调用费用（有免费额度）

### 2. 本地模型模式

在本地运行千问Plus模型。

**优点**:
- 完全离线
- 无 API 调用费用
- 隐私性好

**缺点**:
- 需要下载大模型
- 需要较高配置的 GPU（建议 16GB+ 显存）

## 快速开始

### 安装依赖

```bash
cd qwen_stroke_extractor
pip install -r requirements.txt
```

### API 模式使用

1. 注册阿里云账号并获取 API Key
   - 访问: https://dashscope.aliyun.com/
   - 创建 API Key

2. 运行示例:

```python
from qwen_stroke_extractor.extractor import create_extractor

# 创建提取器（使用 API 模式）
extractor = create_extractor(
    use_api=True,
    api_key="your-api-key-here"
)

# 提取笔画
result = extractor.extract("path/to/image.png")

# 保存为与原项目兼容的格式
extractor.save_result(result, "output.npy", format="npy")
```

### 本地模型模式使用

1. 下载千问Plus模型:
   ```bash
   # 从 HuggingFace 下载
   git lfs install
   git clone https://huggingface.co/Qwen/Qwen-VL-Plus
   ```

2. 运行示例:

```python
from qwen_stroke_extractor.extractor import create_extractor

# 创建提取器（使用本地模型）
extractor = create_extractor(
    model_path="path/to/Qwen-VL-Plus"
)

# 提取笔画
result = extractor.extract("path/to/image.png")

# 保存结果
extractor.save_result(result, "output.npy", format="npy")
```

## 命令行使用

```bash
# 使用 API 模式
python examples.py path/to/image.png --use-api --api-key YOUR_KEY --visualize

# 使用本地模型
python examples.py path/to/image.png --model-path ./Qwen-VL-Plus --visualize
```

## 输出格式

### 参数化格式 (params)

与原 seq_extract 兼容的格式: `(N, 7)`

```python
# 每个笔画的参数:
[x1, y1, x2, y2, width, pressure, eos]
#  [0-1]  [0-1]  [0-1] [0-1] [0-1] [0-1]  [0-1]
```

### Stroke3 格式

```python
# 每个点的参数:
[x, y, pen_state]
# [0-1] [0-1] 0=继续, 1=抬笔
```

### JSON 格式

完整的结构化数据，包含所有笔画信息。

## 与原 CalliRewrite 集成

```python
# 加载提取的笔画
import numpy as np
from qwen_stroke_extractor.extractor import create_extractor

# 提取笔画
extractor = create_extractor(...)
result = extractor.extract("calligraphy.png")

# 转换为原项目格式
stroke_params = extractor.to_numpy(result, output_format="params")

# 保存为 .npy
np.save("strokes.npy", stroke_params)

# 现在可以直接用在 rl_finetune 模块中
```

## 提示词定制

可以通过自定义提示词来优化提取效果:

```python
custom_prompt = """请分析这幅书法图像...（你的提示词）"""

result = extractor.extract(
    "image.png",
    prompt=custom_prompt
)
```

## 注意事项

1. **图像质量**: 建议使用清晰、对比度高的书法图像
2. **书写顺序**: 模型会尽量按书写顺序识别，但可能需要后处理调整
3. **参数调整**: 可以通过调整提示词来优化特定风格的书法

## 故障排除

### 显存不足 (OOM)

如果使用本地模型遇到 OOM:
- 使用更小的模型（如 Qwen-VL）
- 使用 CPU 模式（速度较慢）
- 使用 API 模式

### API 调用失败

检查:
- API Key 是否正确
- 网络连接是否正常
- 是否有调用额度

## 许可证

与 CalliRewrite 项目保持一致。

## 参考链接

- 千问模型: https://github.com/QwenLM/Qwen-VL
- 阿里云 DashScope: https://dashscope.aliyun.com/
- CalliRewrite: ../README.md

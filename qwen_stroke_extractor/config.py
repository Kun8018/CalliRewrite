"""
千问Plus笔画提取器配置
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QwenConfig:
    """
    千问模型配置
    """
    # 模型名称或路径
    model_name: str = "Qwen/Qwen-VL-Plus"

    # 是否使用本地模型
    use_local: bool = True

    # 本地模型路径（如果 use_local=True 且模型不在默认位置）
    model_path: Optional[str] = None

    # 设备
    device: str = "auto"  # "auto" 自动选择，"cpu" 或 "cuda"

    # 最大生成 token 数
    max_new_tokens: int = 2048

    # 温度
    temperature: float = 0.1

    # 束搜索大小（0 表示不使用束搜索）
    num_beams: int = 0

    # 是否使用贪婪解码
    do_sample: bool = False


@dataclass
class ExtractionConfig:
    """
    笔画提取配置
    """
    # 是否可视化结果
    visualize: bool = True

    # 可视化输出目录
    viz_dir: str = "outputs/visualization"

    # 输出格式
    output_formats: List[str] = ("npy", "json")

    # 坐标归一化
    normalized: bool = True

    # 是否保存原始比例图像
    save_original: bool = False


@dataclass
class ApiConfig:
    """
    API 配置
    """
    # 是否使用 API
    use_api: bool = False

    # API 密钥
    api_key: Optional[str] = None

    # API 基础 URL（阿里云 DashScope 为例）
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    # 模型版本
    api_model: str = "qwen-vl-plus"

    # API 调用超时（秒）
    timeout: int = 60


@dataclass
class GlobalConfig:
    """
    全局配置
    """
    # 千问模型配置
    qwen: QwenConfig = QwenConfig()

    # 提取配置
    extraction: ExtractionConfig = ExtractionConfig()

    # API 配置
    api: ApiConfig = ApiConfig()


# 默认配置
DEFAULT_CONFIG = GlobalConfig()


def load_config(config_file: str) -> GlobalConfig:
    """
    从文件加载配置

    Args:
        config_file: 配置文件路径

    Returns:
        配置对象
    """
    import json
    from typing import Dict

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # 简单的配置加载
    qwen_config = QwenConfig(**config_data.get("qwen", {}))
    extraction_config = ExtractionConfig(**config_data.get("extraction", {}))
    api_config = ApiConfig(**config_data.get("api", {}))

    return GlobalConfig(
        qwen=qwen_config,
        extraction=extraction_config,
        api=api_config
    )


def save_config(config: GlobalConfig, config_file: str):
    """
    保存配置到文件

    Args:
        config: 配置对象
        config_file: 输出文件路径
    """
    import json

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            "qwen": config.qwen.__dict__,
            "extraction": config.extraction.__dict__,
            "api": config.api.__dict__
        }, f, ensure_ascii=False, indent=2)

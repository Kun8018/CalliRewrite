"""
基于千问Plus的图像到笔画提取器
"""
import os
import json
import base64
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@dataclass
class StrokePoint:
    """笔画点数据结构"""
    x: float
    y: float
    pressure: float = 1.0


@dataclass
class Stroke:
    """笔画数据结构"""
    points: List[StrokePoint]
    width: float = 2.0
    is_complete: bool = True


@dataclass
class StrokeExtractionResult:
    """笔画提取结果"""
    strokes: List[Stroke]
    image_size: Tuple[int, int]
    normalized: bool = True


class QwenStrokeExtractor:
    """
    基于千问Plus的图像到笔画提取器

    使用千问多模态大模型分析书法图像，
    识别笔画并输出参数化的笔画序列
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "Qwen/Qwen-VL-Plus",
        device: str = None,
        use_api: bool = False,
        api_key: Optional[str] = None
    ):
        """
        初始化提取器

        Args:
            model_path: 本地模型路径
            model_name: 模型名称
            device: 计算设备 ('cpu', 'cuda', 'cuda:0')
            use_api: 是否使用API方式
            api_key: API密钥
        """
        self.model_path = model_path
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = None
        self.tokenizer = None
        self.processor = None

        # 提示词模板
        self.prompt_template = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认的提示词模板"""
        return """请分析这幅书法图像，识别其中的笔画。

要求：
1. 按书写顺序列出所有笔画
2. 每个笔画用起点和终点表示
3. 坐标归一化到[0, 1]范围，左上角为(0,0)，右下角为(1,1)
4. 估计每个笔画的宽度（归一化值）

请严格按以下JSON格式输出，不要包含其他文字：

{
  "image_size": [width, height],
  "strokes": [
    {
      "id": 0,
      "start_point": [x1, y1],
      "end_point": [x2, y2],
      "width": 0.02,
      "type": "horizontal"
    }
  ]
}

笔画类型可以是：horizontal（横）、vertical（竖）、left-falling（撇）、right-falling（捺）、dot（点）、hook（钩）、other（其他）
"""

    def _load_model_local(self):
        """加载本地千问模型"""
        try:
            from transformers import AutoModelForVisionAndLanguageGeneration, AutoProcessor

            print(f"加载千问模型: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path or self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForVisionAndLanguageGeneration.from_pretrained(
                self.model_path or self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.device
            )

            self.model.eval()
            print("模型加载完成")

        except ImportError:
            print("需要安装 transformers 库")
            print("请运行: pip install transformers torch torchvision pillow")
            raise
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("\n提示：如果本地加载失败，可以尝试使用API模式")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """调用千问API（阿里云DashScope）"""
        import requests
        import json
        import base64

        print("调用千问Plus API...")

        # 读取图像并编码
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 构建请求体
        payload = {
            "model": "qwen-vl-plus",
            "input": {
                "prompt": prompt,
                "image": f"data:image/png;base64,{image_data}"
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }

        # API配置
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 发送请求
        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multi-modal-generation/generation",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                if "output" in data and "text" in data["output"]:
                    return data["output"]["text"]
                else:
                    print(f"API响应格式错误: {data}")
                    raise Exception("API响应格式错误")
            else:
                print(f"API请求失败: {response.status_code} - {response.text}")
                raise Exception(f"API请求失败: {response.status_code}")

        except Exception as e:
            print(f"API调用出错: {e}")
            raise

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _parse_response(self, response: str) -> StrokeExtractionResult:
        """解析模型返回的JSON响应"""
        try:
            # 尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("无法找到JSON格式的响应")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # 转换为我们的数据结构
            strokes = []
            for stroke_data in data.get('strokes', []):
                start_point = StrokePoint(
                    x=stroke_data['start_point'][0],
                    y=stroke_data['start_point'][1],
                    pressure=1.0
                )
                end_point = StrokePoint(
                    x=stroke_data['end_point'][0],
                    y=stroke_data['end_point'][1],
                    pressure=1.0
                )

                # 简单地用起点和终点创建笔画
                # 实际应用中可以添加更多中间点
                stroke = Stroke(
                    points=[start_point, end_point],
                    width=stroke_data.get('width', 0.02),
                    is_complete=True
                )
                strokes.append(stroke)

            image_size = tuple(data.get('image_size', [256, 256]))

            return StrokeExtractionResult(
                strokes=strokes,
                image_size=image_size,
                normalized=True
            )

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"响应内容: {response}")
            raise
        except Exception as e:
            print(f"解析响应时出错: {e}")
            raise

    def extract(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 2048
    ) -> StrokeExtractionResult:
        """
        从图像中提取笔画

        Args:
            image_path: 输入图像路径
            prompt: 自定义提示词（可选）
            max_new_tokens: 最大生成token数

        Returns:
            笔画提取结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 获取图像尺寸
        with Image.open(image_path) as img:
            orig_size = img.size

        if self.use_api:
            # API模式
            response = self._call_api(image_path, prompt or self.prompt_template)
        else:
            # 本地模型模式
            if self.model is None:
                self._load_model_local()

            # 准备输入
            query = prompt or self.prompt_template

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_path},
                        {"text": query}
                    ]
                }
            ]

            # 处理输入
            inputs = self.processor(messages, return_tensors="pt")
            inputs = inputs.to(self.device)

            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1
                )

            # 解码输出
            response = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            # 提取助手回复
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

        # 解析响应
        result = self._parse_response(response)

        # 更新图像尺寸（使用实际尺寸）
        result.image_size = orig_size

        return result

    def to_numpy(
        self,
        result: StrokeExtractionResult,
        output_format: str = "stroke3"
    ) -> np.ndarray:
        """
        将提取结果转换为numpy数组

        Args:
            result: 笔画提取结果
            output_format: 输出格式
                - "stroke3": (N, 3) 格式 [x, y, pen_state]
                - "params": (N, 7) 格式 [x1, y1, x2, y2, width, pressure, eos]

        Returns:
            numpy数组
        """
        if output_format == "stroke3":
            # 转换为stroke3格式
            points = []
            for stroke in result.strokes:
                for i, point in enumerate(stroke.points):
                    pen_state = 1 if i == len(stroke.points) - 1 else 0
                    points.append([point.x, point.y, pen_state])

            return np.array(points, dtype=np.float32)

        elif output_format == "params":
            # 转换为参数格式
            params = []
            for stroke in result.strokes:
                if len(stroke.points) >= 2:
                    start = stroke.points[0]
                    end = stroke.points[-1]
                    params.append([
                        start.x, start.y,
                        end.x, end.y,
                        stroke.width,
                        (start.pressure + end.pressure) / 2,
                        0.0  # eos (will be set for last stroke)
                    ])

            if params:
                params[-1][6] = 1.0  # 设置最后一个笔画的eos标志

            return np.array(params, dtype=np.float32)

        else:
            raise ValueError(f"未知的输出格式: {output_format}")

    def save_result(
        self,
        result: StrokeExtractionResult,
        output_path: str,
        format: str = "npy"
    ):
        """
        保存提取结果

        Args:
            result: 笔画提取结果
            output_path: 输出路径
            format: 保存格式 ("npy", "json")
        """
        if format == "npy":
            # 保存为与原项目兼容的npy格式
            data = self.to_numpy(result, output_format="params")
            np.save(output_path, data)
        elif format == "json":
            # 保存为JSON格式
            data = {
                "image_size": result.image_size,
                "normalized": result.normalized,
                "strokes": [
                    {
                        "points": [{"x": p.x, "y": p.y, "pressure": p.pressure} for p in s.points],
                        "width": s.width,
                        "is_complete": s.is_complete
                    }
                    for s in result.strokes
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"未知的保存格式: {format}")

    def visualize_result(
        self,
        result: StrokeExtractionResult,
        output_path: str,
        background_image: str = None
    ):
        """
        可视化笔画提取结果

        Args:
            result: 笔画提取结果
            output_path: 输出图像路径
            background_image: 背景图像（可选）
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection
        import numpy as np

        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制背景图像
        if background_image and os.path.exists(background_image):
            from PIL import Image
            img = Image.open(background_image)
            ax.imshow(img, alpha=0.3)

        # 设置坐标轴
        ax.set_xlim(0, result.image_size[0])
        ax.set_ylim(result.image_size[1], 0)  # 翻转Y轴
        ax.set_aspect('equal')

        # 绘制笔画
        colors = plt.cm.rainbow(np.linspace(0, 1, len(result.strokes)))

        for i, stroke in enumerate(result.strokes):
            if len(stroke.points) >= 2:
                # 转换坐标（如果归一化了）
                if result.normalized:
                    points = np.array([
                        [p.x * result.image_size[0], p.y * result.image_size[1]]
                        for p in stroke.points
                    ])
                else:
                    points = np.array([[p.x, p.y] for p in stroke.points])

                # 绘制线条
                ax.plot(points[:, 0], points[:, 1],
                       color=colors[i], linewidth=stroke.width * 2,
                       alpha=0.8, label=f"Stroke {i+1}")

                # 绘制起点和终点
                ax.scatter(points[0, 0], points[0, 1], color=colors[i], s=50, marker='o', zorder=10)
                ax.scatter(points[-1, 0], points[-1, 1], color=colors[i], s=50, marker='x', zorder=10)

                # 标注笔画序号
                mid_point = points[len(points) // 2]
                ax.text(mid_point[0], mid_point[1], f"{i+1}",
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        ax.set_title(f"笔画提取结果 (共 {len(result.strokes)} 个笔画)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图像
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {output_path}")
        plt.close()


def create_extractor(
    model_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen-VL-Plus",
    use_api: bool = False,
    api_key: Optional[str] = None
) -> QwenStrokeExtractor:
    """
    创建笔画提取器实例

    Args:
        model_path: 本地模型路径
        model_name: 模型名称
        use_api: 是否使用API方式
        api_key: API密钥

    Returns:
        笔画提取器实例
    """
    return QwenStrokeExtractor(
        model_path=model_path,
        model_name=model_name,
        use_api=use_api,
        api_key=api_key
    )

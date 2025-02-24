"""
模型配置文件
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os

@dataclass
class ModelConfig:
    """模型配置"""
    # 模型基本配置
    model_path: str = "models/llava_sbs"  # 本地模型路径
    device: str = "cuda"
    
    # 输入配置
    max_length: int = 4096
    image_size: Tuple[int, int] = (224, 224)
    
    # 推理配置
    batch_size: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 性能配置
    use_fp16: bool = True
    use_cache: bool = True
    num_threads: int = 4
    
    def __post_init__(self):
        """初始化后的处理"""
        # 验证模型路径
        model_path = os.path.join(os.getcwd(), self.model_path)
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")

# 默认配置
DEFAULT_CONFIG = ModelConfig()

def load_config() -> ModelConfig:
    """加载模型配置"""
    return DEFAULT_CONFIG
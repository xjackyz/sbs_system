from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """配置类"""
    # 必需的配置
    model_path: str
    vision_model_path: str = "openai/clip-vit-large-patch14-336"  # CLIP视觉模型路径
    device: str = "cuda"
    
    # 模型参数
    batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 4
    
    # 默认提示文本
    default_prompt: str = """请分析这张图表并关注：
1. SBS序列的完整性和有效性
2. 关键点位的位置和重要性
3. 交易信号的生成和确认
4. 市场结构和趋势状态"""
    
    # 缓存设置
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # 性能优化
    use_flash_attention: bool = True
    use_bettertransformer: bool = True
    use_cuda_graph: bool = True
    
    # SSL和网络设置
    verify_ssl: bool = True
    use_mirror: bool = True
    mirror_url: str = "https://hf-mirror.com"
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.cache_dir is None:
            self.cache_dir = ".cache" 
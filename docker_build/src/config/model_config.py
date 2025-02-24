"""
模型配置
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ModelConfig:
    """模型配置"""
    
    # 基础配置
    model_dir: str = field(default="models")  # 模型文件目录
    config_dir: str = field(default="configs/models")  # 模型配置目录
    cache_dir: str = field(default="cache/models")  # 模型缓存目录
    
    # 资源限制
    max_gpu_memory: float = field(default=8192)  # 最大GPU内存使用(MB)
    max_model_size: float = field(default=4096)  # 最大模型大小(MB)
    max_batch_size: int = field(default=32)  # 最大批处理大小
    
    # 性能配置
    use_fp16: bool = field(default=True)  # 是否使用半精度
    use_cuda: bool = field(default=True)  # 是否使用CUDA
    num_workers: int = field(default=4)  # 数据加载线程数
    
    # 监控配置
    monitor_interval: float = field(default=30.0)  # 监控间隔(秒)
    log_level: str = field(default="INFO")  # 日志级别
    
    # 模型特定配置
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置默认设备
        if not self.use_cuda:
            self.device = "cpu"
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
@dataclass
class ImageModelConfig:
    """图像模型配置"""
    
    # 输入配置
    input_size: tuple = field(default=(224, 224))  # 输入图像大小
    input_channels: int = field(default=3)  # 输入通道数
    normalize_mean: tuple = field(default=(0.485, 0.456, 0.406))  # 归一化均值
    normalize_std: tuple = field(default=(0.229, 0.224, 0.225))  # 归一化标准差
    
    # 数据增强
    use_augmentation: bool = field(default=True)  # 是否使用数据增强
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        'random_crop': True,
        'random_flip': True,
        'color_jitter': True
    })
    
    # 模型架构
    backbone: str = field(default="resnet50")  # 骨干网络
    pretrained: bool = field(default=True)  # 是否使用预训练权重
    freeze_backbone: bool = field(default=False)  # 是否冻结骨干网络
    
@dataclass
class TextModelConfig:
    """文本模型配置"""
    
    # 输入配置
    max_length: int = field(default=512)  # 最大序列长度
    vocab_size: int = field(default=30000)  # 词表大小
    padding: str = field(default="max_length")  # 填充策略
    truncation: bool = field(default=True)  # 是否截断
    
    # 模型架构
    model_type: str = field(default="bert")  # 模型类型
    num_layers: int = field(default=12)  # 层数
    hidden_size: int = field(default=768)  # 隐藏层大小
    num_heads: int = field(default=12)  # 注意力头数
    
    # 训练配置
    dropout: float = field(default=0.1)  # Dropout比率
    attention_dropout: float = field(default=0.1)  # 注意力Dropout比率
    
@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基础训练参数
    epochs: int = field(default=100)  # 训练轮数
    batch_size: int = field(default=32)  # 批处理大小
    learning_rate: float = field(default=1e-4)  # 学习率
    weight_decay: float = field(default=1e-2)  # 权重衰减
    
    # 优化器配置
    optimizer: str = field(default="adam")  # 优化器类型
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    })
    
    # 学习率调度
    scheduler: str = field(default="cosine")  # 学习率调度器类型
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'warmup_steps': 1000,
        'num_cycles': 0.5
    })
    
    # 训练控制
    gradient_clip: float = field(default=1.0)  # 梯度裁剪
    early_stopping_patience: int = field(default=10)  # 早停耐心值
    validation_interval: int = field(default=1)  # 验证间隔(轮) 
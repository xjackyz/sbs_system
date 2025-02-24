"""
统一配置系统
"""
import os
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BaseConfig:
    """基础配置"""
    
    # 系统基本配置
    run_mode: str = "production"  # production, development, test
    debug: bool = False
    log_level: str = "INFO"
    use_gpu: bool = True
    num_workers: int = 4
    
    # 目录配置
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    output_dir: str = "output"
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证运行模式
        valid_modes = ["production", "development", "test"]
        if self.run_mode not in valid_modes:
            raise ValueError(f"无效的运行模式: {self.run_mode}")
            
        # 验证日志级别
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"无效的日志级别: {self.log_level}")
            
        # 验证工作线程数
        if self.num_workers < 1:
            raise ValueError(f"无效的工作线程数: {self.num_workers}")
            
        # GPU可用性检查
        if self.use_gpu:
            self.use_gpu = torch.cuda.is_available()
            
        # 创建必要的目录
        for directory in [self.data_dir, self.model_dir, self.log_dir, 
                         self.cache_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class ModelConfig:
    """模型配置"""
    
    # 模型基本配置
    model_path: str = "models/llava-sbs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        """初始化后处理"""
        # 验证模型路径
        if not os.path.exists(self.model_path):
            raise ValueError(f"模型路径不存在: {self.model_path}")
            
        # 验证其他参数
        if self.num_threads < 1:
            raise ValueError("num_threads必须大于0")
        if not (0 < self.temperature <= 1):
            raise ValueError("temperature必须在(0,1]范围内")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p必须在(0,1]范围内")

@dataclass
class InputConfig:
    """输入系统配置"""
    
    # 历史数据配置
    historical_data_dir: str = "data/historical"
    historical_cache_dir: str = "cache/historical"
    historical_batch_size: int = 1000
    historical_cache_size: int = 10000
    
    # 截图配置
    capture_save_dir: str = "data/captures"
    capture_width: int = 1920
    capture_height: int = 1080
    capture_quality: int = 90
    capture_interval: int = 60
    
    # Discord配置
    discord_token: str = os.getenv("DISCORD_TOKEN", "")
    discord_prefix: str = "!"
    discord_admin_roles: List[str] = field(default_factory=lambda: ["admin", "moderator"])
    discord_save_dir: str = "data/discord"
    
    # API配置
    api_base_url: str = "https://api.binance.com"
    api_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    api_interval: str = "1m"
    api_update_interval: int = 1
    api_buffer_size: int = 1000
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        for directory in [self.historical_data_dir, self.historical_cache_dir,
                         self.capture_save_dir, self.discord_save_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class MonitorConfig:
    """监控配置"""
    
    # 资源阈值
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    gpu_threshold: float = 75.0
    
    # 监控间隔
    check_interval: int = 60
    log_interval: int = 300
    alert_cooldown: int = 1800
    
    # Discord配置
    discord_webhook_monitor: str = os.getenv("DISCORD_WEBHOOK_MONITOR", "")
    discord_webhook_signal: str = os.getenv("DISCORD_WEBHOOK_SIGNAL", "")
    discord_webhook_debug: str = os.getenv("DISCORD_WEBHOOK_DEBUG", "")
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证阈值范围
        thresholds = {
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold,
            'disk_threshold': self.disk_threshold,
            'gpu_threshold': self.gpu_threshold
        }
        
        for name, value in thresholds.items():
            if not 0 <= value <= 100:
                raise ValueError(f"{name}必须在[0,100]范围内")
            
        # 验证时间间隔
        if self.check_interval < 1:
            raise ValueError("check_interval必须大于0")
        if self.log_interval < self.check_interval:
            raise ValueError("log_interval必须大于等于check_interval")
        if self.alert_cooldown < self.check_interval:
            raise ValueError("alert_cooldown必须大于等于check_interval")

@dataclass
class SystemConfig:
    """系统总配置"""
    
    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    input: InputConfig = field(default_factory=InputConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'base': self.base.__dict__,
            'model': self.model.__dict__,
            'input': self.input.__dict__,
            'monitor': self.monitor.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SystemConfig':
        """从字典创建配置"""
        return cls(
            base=BaseConfig(**config_dict.get('base', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            input=InputConfig(**config_dict.get('input', {})),
            monitor=MonitorConfig(**config_dict.get('monitor', {}))
        )

def load_config() -> SystemConfig:
    """加载系统配置"""
    return SystemConfig() 
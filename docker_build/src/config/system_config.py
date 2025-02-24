"""
系统配置模块
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class SystemConfig:
    """系统配置类"""
    run_mode: str = 'production'  # production, development, test
    debug: bool = False
    log_level: str = 'INFO'
    use_gpu: bool = True
    num_workers: int = 4
    data_dir: str = 'data'
    model_dir: str = 'models'
    log_dir: str = 'logs'
    cache_dir: str = 'cache'
    temp_dir: str = 'temp'
    
    def __post_init__(self):
        """初始化后的处理"""
        # 验证运行模式
        valid_modes = ['production', 'development', 'test']
        if self.run_mode not in valid_modes:
            raise ValueError(f"Invalid run_mode: {self.run_mode}. Must be one of {valid_modes}")
            
        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}")
            
        # 验证工作线程数
        if self.num_workers < 1:
            raise ValueError(f"Invalid num_workers: {self.num_workers}. Must be >= 1")
            
        # 创建必要的目录
        for dir_path in [self.data_dir, self.model_dir, self.log_dir,
                        self.cache_dir, self.temp_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'run_mode': self.run_mode,
            'debug': self.debug,
            'log_level': self.log_level,
            'use_gpu': self.use_gpu,
            'num_workers': self.num_workers,
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'log_dir': self.log_dir,
            'cache_dir': self.cache_dir,
            'temp_dir': self.temp_dir
        }
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SystemConfig':
        """从字典创建配置"""
        return cls(**config_dict)
        
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config key: {key}")
                
        # 重新验证配置
        self._validate_config() 
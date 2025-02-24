"""
数据配置文件
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径配置
    data_root: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    validation_data_dir: str = "data/validation"
    
    # 数据处理配置
    batch_size: int = 32
    num_workers: int = 4
    cache_size: int = 1000
    
    # 图表配置
    chart_size: Tuple[int, int] = (800, 600)
    time_periods: List[str] = None
    symbols: List[str] = None
    
    # 数据验证配置
    min_data_points: int = 100
    max_missing_ratio: float = 0.01
    outlier_std_threshold: float = 3.0
    
    # 数据增强配置
    use_augmentation: bool = True
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    def __post_init__(self):
        if self.time_periods is None:
            self.time_periods = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.symbols is None:
            self.symbols = ["NQ"]

# 默认配置
DEFAULT_CONFIG = DataConfig()

def load_config() -> DataConfig:
    """加载数据配置"""
    return DEFAULT_CONFIG 
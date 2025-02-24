"""
系统配置文件
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class SystemConfig:
    """系统配置"""
    # 系统基本配置
    debug: bool = False
    log_level: str = "INFO"
    use_gpu: bool = True
    num_workers: int = 4
    
    # 监控配置
    monitor_interval: int = 60  # 秒
    memory_threshold: float = 0.9  # 内存使用率阈值
    cpu_threshold: float = 0.8    # CPU使用率阈值
    gpu_threshold: float = 0.9    # GPU使用率阈值
    disk_threshold: float = 0.9   # 磁盘使用率阈值
    
    # 日志配置
    log_dir: str = "logs"
    max_log_size: int = 100  # MB
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Discord配置
    discord_webhook_monitor: Optional[str] = None
    discord_webhook_signal: Optional[str] = None
    discord_webhook_debug: Optional[str] = None
    discord_username: str = "SBS Trading Bot"
    discord_avatar_url: Optional[str] = None
    
    # 错误处理
    max_retries: int = 3
    retry_delay: int = 5  # 秒
    error_threshold: int = 10
    error_cooldown: int = 300  # 秒
    
    # 性能配置
    profile_code: bool = False
    memory_tracking: bool = True
    use_async: bool = True
    queue_size: int = 1000
    
    # 安全配置
    enable_ssl: bool = False
    api_key: Optional[str] = None
    allowed_ips: List[str] = None
    
    # 系统维护
    auto_cleanup: bool = True
    cleanup_interval: int = 86400  # 24小时
    max_log_age: int = 7  # 天
    max_cache_size: int = 10  # GB
    
    def __post_init__(self):
        if self.allowed_ips is None:
            self.allowed_ips = ["127.0.0.1"]

# 默认配置
DEFAULT_CONFIG = SystemConfig()

def load_config() -> SystemConfig:
    """加载系统配置"""
    return DEFAULT_CONFIG 
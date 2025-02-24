"""
监控配置模块
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class MonitorConfig:
    """监控配置类"""
    # 资源阈值
    cpu_threshold: float = 80.0  # CPU使用率阈值(%)
    memory_threshold: float = 85.0  # 内存使用率阈值(%)
    disk_threshold: float = 90.0  # 磁盘使用率阈值(%)
    gpu_threshold: float = 75.0  # GPU使用率阈值(%)
    
    # 监控间隔
    check_interval: int = 60  # 检查间隔(秒)
    log_interval: int = 300  # 日志记录间隔(秒)
    alert_cooldown: int = 1800  # 告警冷却时间(秒)
    
    # Discord配置
    discord_webhook_url: Optional[str] = None  # Discord Webhook URL
    
    # 功能开关
    enable_performance_tracking: bool = True  # 启用性能跟踪
    enable_error_tracking: bool = True  # 启用错误跟踪
    enable_resource_tracking: bool = True  # 启用资源跟踪
    
    def __post_init__(self):
        """初始化后的处理"""
        self._validate_config()
        
    def _validate_config(self):
        """验证配置"""
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
            
        # 验证Discord Webhook URL
        if self.discord_webhook_url is not None and not self.discord_webhook_url.startswith('https://discord.com/api/webhooks/'):
            raise ValueError(f"无效的discord_webhook_url: {self.discord_webhook_url}")
            
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold,
            'disk_threshold': self.disk_threshold,
            'gpu_threshold': self.gpu_threshold,
            'check_interval': self.check_interval,
            'log_interval': self.log_interval,
            'alert_cooldown': self.alert_cooldown,
            'discord_webhook_url': self.discord_webhook_url,
            'enable_performance_tracking': self.enable_performance_tracking,
            'enable_error_tracking': self.enable_error_tracking,
            'enable_resource_tracking': self.enable_resource_tracking
        }
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MonitorConfig':
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
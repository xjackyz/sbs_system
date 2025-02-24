"""
输入系统配置
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os

@dataclass
class HistoricalDataConfig:
    """历史数据配置"""
    
    # 数据目录
    data_dir: str = "data/historical"
    
    # 数据加载参数
    batch_size: int = 1000
    cache_size: int = 10000
    
    # 支持的时间周期
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # 缓存配置
    use_cache: bool = True
    cache_dir: str = "cache/historical"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

@dataclass
class CaptureConfig:
    """截图配置"""
    
    # 截图保存目录
    save_dir: str = "data/captures"
    
    # 截图参数
    width: int = 1920
    height: int = 1080
    quality: int = 90
    
    # 截图间隔(秒)
    interval: int = 60
    
    # 重试配置
    max_retries: int = 3
    retry_delay: int = 5
    
    # 浏览器配置
    browser_args: List[str] = field(default_factory=lambda: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-accelerated-2d-canvas",
        "--disable-gpu",
        "--window-size=1920,1080"
    ])
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.save_dir, exist_ok=True)

@dataclass
class DiscordConfig:
    """Discord配置"""
    
    # 机器人配置
    token: str = os.getenv("DISCORD_TOKEN", "")
    prefix: str = "!"
    
    # 权限配置
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "moderator"])
    
    # 命令配置
    cooldown: int = 60
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg"])
    
    # 图片保存配置
    save_dir: str = "data/discord"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.save_dir, exist_ok=True)

@dataclass
class APIConfig:
    """API配置"""
    
    # API端点
    base_url: str = "https://api.binance.com"
    historical_endpoint: str = "/api/v3/klines"
    realtime_endpoint: str = "/api/v3/ticker/24hr"
    
    # 请求配置
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    })
    
    # 交易对配置
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    interval: str = "1m"
    
    # 更新配置
    update_interval: int = 1
    min_update_interval: int = 1
    max_age: int = 5
    buffer_size: int = 1000
    
    # API密钥(如果需要)
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")

@dataclass
class InputSystemConfig:
    """输入系统总配置"""
    
    # 子系统配置
    historical_config: HistoricalDataConfig = field(default_factory=HistoricalDataConfig)
    capture_config: CaptureConfig = field(default_factory=CaptureConfig)
    discord_config: DiscordConfig = field(default_factory=DiscordConfig)
    api_config: APIConfig = field(default_factory=APIConfig)
    
    # 系统配置
    monitor_interval: int = 60
    log_dir: str = "logs/input"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.log_dir, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = InputSystemConfig()

def load_config() -> InputSystemConfig:
    """加载输入系统配置"""
    return DEFAULT_CONFIG 
"""
输入系统基础接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd

class InputSource(ABC):
    """输入源基类"""
    
    def __init__(self):
        """初始化"""
        self.running = False
        self.last_update = None
        
    @abstractmethod
    async def start(self):
        """启动输入源"""
        self.running = True
        
    @abstractmethod
    async def stop(self):
        """停止输入源"""
        self.running = False
        
    @abstractmethod
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """获取数据"""
        pass
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'running': self.running,
            'last_update': self.last_update
        }
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop()

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> bool:
        """验证市场数据"""
        if data is None or data.empty:
            return False
            
        # 检查必需的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # 检查数据类型
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
            
        # 检查数值范围
        if (data[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            return False
            
        return True
        
    @staticmethod
    def validate_image_data(image_data: Dict) -> bool:
        """验证图像数据"""
        if not image_data or not isinstance(image_data, dict):
            return False
            
        required_fields = ['image', 'timestamp', 'symbol']
        if not all(field in image_data for field in required_fields):
            return False
            
        # 检查图像数据
        if not image_data['image'] or not isinstance(image_data['image'], (str, bytes)):
            return False
            
        # 检查时间戳
        if not isinstance(image_data['timestamp'], (str, datetime)):
            return False
            
        return True
        
    @staticmethod
    def validate_discord_data(message_data: Dict) -> bool:
        """验证Discord消息数据"""
        if not message_data or not isinstance(message_data, dict):
            return False
            
        required_fields = ['content', 'author', 'timestamp']
        if not all(field in message_data for field in required_fields):
            return False
            
        # 检查消息内容
        if not message_data['content'] or not isinstance(message_data['content'], str):
            return False
            
        # 检查时间戳
        if not isinstance(message_data['timestamp'], (str, datetime)):
            return False
            
        return True

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def process_market_data(data: pd.DataFrame) -> pd.DataFrame:
        """处理市场数据"""
        if not DataValidator.validate_market_data(data):
            raise ValueError("无效的市场数据")
            
        # 确保时间索引
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data['timestamp'])
            
        # 排序
        data = data.sort_index()
        
        # 删除重复数据
        data = data.drop_duplicates()
        
        # 填充缺失值
        data = data.ffill()
        
        return data
        
    @staticmethod
    def process_image_data(image_data: Dict) -> Dict:
        """处理图像数据"""
        if not DataValidator.validate_image_data(image_data):
            raise ValueError("无效的图像数据")
            
        # 标准化时间戳
        if isinstance(image_data['timestamp'], str):
            image_data['timestamp'] = pd.to_datetime(image_data['timestamp'])
            
        return image_data
        
    @staticmethod
    def process_discord_data(message_data: Dict) -> Dict:
        """处理Discord消息数据"""
        if not DataValidator.validate_discord_data(message_data):
            raise ValueError("无效的Discord消息数据")
            
        # 标准化时间戳
        if isinstance(message_data['timestamp'], str):
            message_data['timestamp'] = pd.to_datetime(message_data['timestamp'])
            
        # 清理消息内容
        message_data['content'] = message_data['content'].strip()
        
        return message_data 
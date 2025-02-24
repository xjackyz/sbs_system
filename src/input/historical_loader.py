"""
历史数据加载器
"""
import os
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from .base import InputSource, DataValidator, DataProcessor
from ..config.input_config import HistoricalDataConfig

class HistoricalDataLoader(InputSource):
    """历史数据加载器"""
    
    def __init__(self, config: Optional[HistoricalDataConfig] = None):
        """初始化"""
        super().__init__()
        self.config = config or HistoricalDataConfig()
        self._setup_cache()
        self.data_cache = {}
        
    def _setup_cache(self):
        """设置缓存目录"""
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
    async def start(self):
        """启动加载器"""
        await super().start()
        
    async def stop(self):
        """停止加载器"""
        await super().stop()
        self.clear_cache()
        
    async def get_data(self, symbol: str, timeframe: str,
                    start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """获取历史数据"""
        try:
            # 加载数据
            df = await self._load_symbol_data(symbol, timeframe, start_date, end_date)
            if df is None:
                return None
                
            # 处理数据
            df = DataProcessor.process_market_data(df)
            
            # 更新状态
            self.last_update = datetime.now()
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'data': df
            }
            
        except Exception as e:
            print(f"加载历史数据出错: {e}")
            return None
            
    async def _load_symbol_data(self, symbol: str, timeframe: str,
                           start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载单个交易对的数据"""
        # 检查缓存
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        try:
            # 构建文件路径
            file_path = os.path.join(
                self.config.data_dir,
                symbol,
                timeframe,
                f"{start_date}_{end_date}.csv"
            )
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return None
                
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 验证数据
            if not DataValidator.validate_market_data(df):
                return None
                
            # 更新缓存
            if len(self.data_cache) >= self.config.cache_size:
                # 删除最旧的缓存
                oldest_key = next(iter(self.data_cache))
                del self.data_cache[oldest_key]
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            print(f"加载{symbol} {timeframe}数据出错: {e}")
            return None
            
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = super().get_status()
        status.update({
            'cache_size': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys())
        })
        return status
        
    def clear_cache(self):
        """清理缓存"""
        self.data_cache.clear()
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return await super().__aenter__()
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await super().__aexit__(exc_type, exc_val, exc_tb) 
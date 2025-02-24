"""
API实时数据输入模块
"""
import os
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
from .base import InputSource, DataValidator, DataProcessor
from ..config.input_config import APIConfig

class APIDataFetcher(InputSource):
    """API数据获取器"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        """初始化"""
        super().__init__()
        self.config = config or APIConfig()
        self.session = None
        self.data_buffer = {}
        self.fetch_task = None
        self.last_updates = {}
        
    async def start(self):
        """启动数据获取器"""
        await super().start()
        self.session = aiohttp.ClientSession()
        self.fetch_task = asyncio.create_task(self._fetch_loop())
        
    async def stop(self):
        """停止数据获取器"""
        if self.fetch_task:
            self.fetch_task.cancel()
            try:
                await self.fetch_task
            except asyncio.CancelledError:
                pass
                
        if self.session:
            await self.session.close()
            self.session = None
            
        await super().stop()
        
    async def _fetch_loop(self):
        """数据获取循环"""
        while self.running:
            try:
                for symbol in self.config.symbols:
                    if self.running:  # 再次检查运行状态
                        await self._fetch_symbol_data(symbol)
                        
                # 等待下一次获取
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"数据获取循环出错: {e}")
                await asyncio.sleep(5)  # 出错后等待一段时间再重试
                
    async def _fetch_symbol_data(self, symbol: str):
        """获取单个交易对的数据"""
        try:
            # 检查更新间隔
            if not self._should_update(symbol):
                return
                
            # 构建请求URL
            url = f"{self.config.base_url}{self.config.endpoints['klines']}"
            
            # 请求参数
            params = {
                'symbol': symbol,
                'interval': self.config.interval,
                'limit': 1000  # 获取最近的1000根K线
            }
            
            # 发送请求
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    # 处理数据
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df = df.astype(float)
                    
                    # 验证数据
                    if not DataValidator.validate_market_data(df):
                        raise ValueError("无效的市场数据")
                        
                    # 更新缓存
                    self._update_buffer(symbol, df)
                    
                    # 更新状态
                    self.last_update = datetime.now()
                    self.last_updates[symbol] = self.last_update
                    
                else:
                    print(f"获取{symbol}数据失败: HTTP {response.status}")
                    
        except Exception as e:
            print(f"获取{symbol}数据出错: {e}")
            
    def _should_update(self, symbol: str) -> bool:
        """检查是否应该更新数据"""
        if symbol not in self.last_updates:
            return True
            
        time_diff = datetime.now() - self.last_updates[symbol]
        return time_diff.total_seconds() >= self.config.update_interval
        
    def _update_buffer(self, symbol: str, data: pd.DataFrame):
        """更新数据缓存"""
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = data
        else:
            # 合并新数据
            self.data_buffer[symbol] = pd.concat([self.data_buffer[symbol], data])
            
            # 删除重复数据
            self.data_buffer[symbol] = self.data_buffer[symbol].loc[~self.data_buffer[symbol].index.duplicated(keep='last')]
            
            # 按时间排序
            self.data_buffer[symbol].sort_index(inplace=True)
            
            # 限制缓存大小
            if len(self.data_buffer[symbol]) > self.config.buffer_size:
                self.data_buffer[symbol] = self.data_buffer[symbol].iloc[-self.config.buffer_size:]
                
    async def get_data(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """获取数据"""
        try:
            if symbol:
                if symbol not in self.data_buffer:
                    return None
                    
                data = self.data_buffer[symbol]
                last_update = self.last_updates.get(symbol)
                
            else:
                # 返回所有数据
                data = self.data_buffer
                last_update = self.last_update
                
            return {
                'data': data,
                'timestamp': last_update,
                'symbols': list(self.data_buffer.keys())
            }
            
        except Exception as e:
            print(f"获取数据出错: {e}")
            return None
            
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = super().get_status()
        status.update({
            'symbols': list(self.data_buffer.keys()),
            'buffer_sizes': {symbol: len(data) for symbol, data in self.data_buffer.items()},
            'last_updates': {symbol: update.strftime("%Y-%m-%d %H:%M:%S") 
                           for symbol, update in self.last_updates.items()}
        })
        return status
        
    def clear_buffer(self, symbol: str = None):
        """清理数据缓存"""
        if symbol:
            if symbol in self.data_buffer:
                del self.data_buffer[symbol]
                del self.last_updates[symbol]
        else:
            self.data_buffer.clear()
            self.last_updates.clear()
            
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop() 
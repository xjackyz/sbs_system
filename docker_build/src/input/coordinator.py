"""
输入系统协调器
"""
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
from .base import InputSource
from .historical_loader import HistoricalDataLoader
from .auto_capture import AutoScreenshotCapture
from .discord_handler import DiscordInputHandler
from .api_data import APIDataFetcher
from ..config.input_config import InputConfig

class InputCoordinator:
    """输入系统协调器"""
    
    def __init__(self, config: Optional[InputConfig] = None):
        """初始化"""
        self.config = config or InputConfig()
        self.running = False
        self.monitor_task = None
        
        # 初始化各个输入源
        self.historical_loader = HistoricalDataLoader(self.config.historical)
        self.auto_capture = AutoScreenshotCapture(self.config.capture)
        self.discord_handler = DiscordInputHandler(self.config.discord)
        self.api_fetcher = APIDataFetcher(self.config.api)
        
        # 存储所有输入源
        self.sources: Dict[str, InputSource] = {
            'historical': self.historical_loader,
            'capture': self.auto_capture,
            'discord': self.discord_handler,
            'api': self.api_fetcher
        }
        
    async def start(self):
        """启动所有输入源"""
        if self.running:
            return
            
        self.running = True
        print("正在启动输入系统...")
        
        # 启动所有输入源
        for name, source in self.sources.items():
            try:
                await source.start()
                print(f"已启动 {name} 输入源")
            except Exception as e:
                print(f"启动 {name} 输入源失败: {e}")
                
        # 启动监控任务
        self.monitor_task = asyncio.create_task(self._monitor_status())
        print("输入系统启动完成")
        
    async def stop(self):
        """停止所有输入源"""
        if not self.running:
            return
            
        self.running = False
        print("正在停止输入系统...")
        
        # 取消监控任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # 停止所有输入源
        for name, source in self.sources.items():
            try:
                await source.stop()
                print(f"已停止 {name} 输入源")
            except Exception as e:
                print(f"停止 {name} 输入源失败: {e}")
                
        print("输入系统已停止")
        
    async def get_historical_data(self, symbol: str, timeframe: str,
                              start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """获取历史数据"""
        try:
            return await self.historical_loader.get_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return None
            
    async def get_latest_data(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """获取最新数据"""
        try:
            return await self.api_fetcher.get_data(symbol)
        except Exception as e:
            print(f"获取最新数据失败: {e}")
            return None
            
    async def capture_chart(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """捕获图表"""
        try:
            # 设置捕获参数
            self.auto_capture.config.symbols = [symbol]
            self.auto_capture.config.timeframes = [timeframe]
            
            # 执行捕获
            return await self.auto_capture.get_data()
            
        except Exception as e:
            print(f"捕获图表失败: {e}")
            return None
            
    async def _monitor_status(self):
        """监控状态"""
        while self.running:
            try:
                # 检查每个输入源的状态
                for name, source in self.sources.items():
                    status = source.get_status()
                    
                    # 检查运行状态
                    if not status['running']:
                        print(f"警告: {name} 输入源已停止运行")
                        
                    # 检查数据新鲜度
                    if status.get('last_update'):
                        age = datetime.now() - status['last_update']
                        if age.total_seconds() > self.config.max_data_age:
                            print(f"警告: {name} 输入源数据已过期")
                            
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                print(f"监控状态出错: {e}")
                await asyncio.sleep(5)
                
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'running': self.running,
            'sources': {}
        }
        
        # 获取每个输入源的状态
        for name, source in self.sources.items():
            try:
                source_status = source.get_status()
                status['sources'][name] = source_status
            except Exception as e:
                status['sources'][name] = {'error': str(e)}
                
        return status
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop() 
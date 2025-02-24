"""
自动截图模块
"""
import os
import asyncio
import hashlib
from typing import Dict, Optional, Any
from datetime import datetime
import pyppeteer
from PIL import Image
import io
from .base import InputSource, DataValidator, DataProcessor
from ..config.input_config import CaptureConfig

class AutoScreenshotCapture(InputSource):
    """自动截图类"""
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """初始化"""
        super().__init__()
        self.config = config or CaptureConfig()
        self.browser = None
        self.page = None
        self.capture_task = None
        self.last_captures = {}
        
    async def start(self):
        """启动截图服务"""
        await super().start()
        await self._setup()
        self.capture_task = asyncio.create_task(self._capture_loop())
        
    async def stop(self):
        """停止截图服务"""
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
                
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
            
        await super().stop()
        
    async def _setup(self):
        """设置浏览器环境"""
        # 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 启动浏览器
        self.browser = await pyppeteer.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        # 创建页面
        self.page = await self.browser.newPage()
        await self.page.setViewport({
            'width': self.config.width,
            'height': self.config.height
        })
        
    async def _capture_loop(self):
        """截图循环"""
        while self.running:
            try:
                for symbol in self.config.symbols:
                    for timeframe in self.config.timeframes:
                        if self.running:  # 再次检查运行状态
                            await self._capture_chart(symbol, timeframe)
                            
                # 等待下一次截图
                await asyncio.sleep(self.config.interval)
                
            except Exception as e:
                print(f"截图循环出错: {e}")
                await asyncio.sleep(5)  # 出错后等待一段时间再重试
                
    async def _capture_chart(self, symbol: str, timeframe: str):
        """捕获单个图表"""
        try:
            # 导航到图表页面
            chart_url = self._get_chart_url(symbol, timeframe)
            await self.page.goto(chart_url, waitUntil='networkidle0')
            
            # 等待图表加载
            await asyncio.sleep(2)
            
            # 设置时间周期
            await self._set_timeframe(timeframe)
            
            # 截图
            screenshot = await self.page.screenshot()
            
            # 检查图片质量
            if not self._check_image_quality(screenshot):
                print(f"图片质量检查失败: {symbol} {timeframe}")
                return
                
            # 生成图片哈希
            image_hash = self._calculate_image_hash(screenshot)
            
            # 检查是否重复
            if self._is_duplicate(symbol, timeframe, image_hash):
                return
                
            # 保存图片
            timestamp = datetime.now()
            filename = f"{symbol}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.config.save_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(screenshot)
                
            # 更新状态
            self.last_update = timestamp
            self.last_captures[(symbol, timeframe)] = {
                'hash': image_hash,
                'timestamp': timestamp,
                'filepath': filepath
            }
            
        except Exception as e:
            print(f"截图失败 {symbol} {timeframe}: {e}")
            
    def _get_chart_url(self, symbol: str, timeframe: str) -> str:
        """获取图表URL"""
        # 这里需要根据实际使用的交易平台修改
        return f"https://tradingview.com/chart/?symbol={symbol}"
        
    async def _set_timeframe(self, timeframe: str):
        """设置时间周期"""
        # 这里需要根据实际使用的交易平台修改
        try:
            # 示例: 点击时间周期选择器并选择指定周期
            await self.page.click('#timeframe-selector')
            await self.page.click(f'#{timeframe}-option')
        except Exception as e:
            print(f"设置时间周期失败: {e}")
            
    def _check_image_quality(self, image_bytes: bytes) -> bool:
        """检查图片质量"""
        try:
            # 打开图片
            image = Image.open(io.BytesIO(image_bytes))
            
            # 检查尺寸
            if image.size != (self.config.width, self.config.height):
                return False
                
            # 检查是否全黑或全白
            extrema = image.convert('L').getextrema()
            if extrema[0] == extrema[1]:  # 所有像素值相同
                return False
                
            return True
            
        except Exception as e:
            print(f"图片质量检查出错: {e}")
            return False
            
    def _calculate_image_hash(self, image_bytes: bytes) -> str:
        """计算图片哈希"""
        return hashlib.md5(image_bytes).hexdigest()
        
    def _is_duplicate(self, symbol: str, timeframe: str, image_hash: str) -> bool:
        """检查是否重复图片"""
        key = (symbol, timeframe)
        if key in self.last_captures:
            last_capture = self.last_captures[key]
            # 如果哈希相同且时间间隔小于配置的间隔，则认为是重复的
            time_diff = datetime.now() - last_capture['timestamp']
            if (last_capture['hash'] == image_hash and 
                time_diff.total_seconds() < self.config.interval):
                return True
        return False
        
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """获取最新数据"""
        if not self.last_captures:
            return None
            
        # 返回最新的截图数据
        latest_capture = max(self.last_captures.values(), 
                           key=lambda x: x['timestamp'])
        
        with open(latest_capture['filepath'], 'rb') as f:
            image_data = f.read()
            
        return {
            'image': image_data,
            'timestamp': latest_capture['timestamp'],
            'filepath': latest_capture['filepath']
        }
        
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = super().get_status()
        status.update({
            'capture_count': len(self.last_captures),
            'symbols': list(set(symbol for symbol, _ in self.last_captures.keys())),
            'timeframes': list(set(tf for _, tf in self.last_captures.keys()))
        })
        return status 
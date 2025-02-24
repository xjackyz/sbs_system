import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import hashlib

from config.config import (
    CHART_WIDTH,
    CHART_HEIGHT,
    CHART_SYMBOLS,
    CHART_TIMEFRAME,
    SCREENSHOT_DIR,
    SCREENSHOT_FORMAT,
    MAX_RETRIES,
    RETRY_DELAY,
    TRADINGVIEW_URL,
    CHART_TEMPLATE,
    MA_SETTINGS
)
from src.utils.logger import setup_logger

logger = setup_logger('screenshot')

class ChartCapture:
    def __init__(self):
        self.setup_driver()
        self.ensure_screenshot_dir()
        self.last_screenshot_hash = {}  # 每个品种的最后截图哈希
        self.last_screenshot_time = {}  # 每个品种的最后截图时间
        self.min_interval = 5  # 最小截图间隔（秒）
        
    def setup_driver(self):
        """设置Chrome浏览器驱动，优化无头模式配置"""
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-software-rasterizer')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--silent')
        chrome_options.add_argument(f'--window-size={CHART_WIDTH},{CHART_HEIGHT}')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_page_load_timeout(30)
        self.driver.set_script_timeout(30)
        
        logger.info("Chrome driver initialized with optimized settings")

    def ensure_screenshot_dir(self):
        """确保截图保存目录存在"""
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)
            logger.info(f"Created screenshot directory: {SCREENSHOT_DIR}")

    def calculate_image_hash(self, image_data):
        """计算图像哈希值"""
        return hashlib.md5(image_data).hexdigest()

    def is_duplicate_screenshot(self, image_data, symbol):
        """检查是否是重复的截图"""
        current_hash = self.calculate_image_hash(image_data)
        current_time = time.time()
        
        if (symbol in self.last_screenshot_hash and 
            self.last_screenshot_hash[symbol] == current_hash and 
            symbol in self.last_screenshot_time and 
            current_time - self.last_screenshot_time[symbol] < self.min_interval):
            return True
            
        self.last_screenshot_hash[symbol] = current_hash
        self.last_screenshot_time[symbol] = current_time
        return False

    def load_chart(self, symbol):
        """加载TradingView图表"""
        try:
            # 构建URL，包含时间框架和指标
            url = f"{TRADINGVIEW_URL}?symbol={symbol}&interval={CHART_TEMPLATE['interval']}"
            self.driver.get(url)
            
            # 等待图表加载
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "chart-container"))
            )
            
            # 添加均线指标
            self._add_indicators()
            
            # 等待图表完全渲染
            time.sleep(2)
            
            logger.info(f"Chart loaded successfully for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load chart for {symbol}: {e}")
            return False

    def _add_indicators(self):
        """添加均线指标"""
        try:
            # 这里需要根据TradingView的具体DOM结构来实现
            # 添加SMA20和SMA200
            for indicator in CHART_TEMPLATE["indicators"]:
                settings = MA_SETTINGS[indicator]
                # 实现添加指标的具体逻辑
                pass
                
        except Exception as e:
            logger.error(f"Failed to add indicators: {e}")

    def capture_chart(self, symbol=None):
        """捕获图表截图"""
        if symbol is None:
            # 如果没有指定品种，轮询所有品种
            results = {}
            for sym in CHART_SYMBOLS:
                results[sym] = self._capture_single_chart(sym)
            return results
        else:
            # 捕获指定品种的图表
            return {symbol: self._capture_single_chart(symbol)}

    def _capture_single_chart(self, symbol):
        """捕获单个品种的图表"""
        try:
            # 加载图表
            if not self.load_chart(symbol):
                return None
                
            # 获取图表元素
            chart_element = self.driver.find_element(By.CLASS_NAME, "chart-container")
            
            # 截取图表
            screenshot = chart_element.screenshot_as_png
            
            # 检查是否是重复的截图
            if self.is_duplicate_screenshot(screenshot, symbol):
                logger.info(f"Duplicate screenshot detected for {symbol}, skipping...")
                return None
            
            # 转换为OpenCV格式并处理
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 检查图像质量
            if not self._check_image_quality(img):
                logger.warning(f"Low quality screenshot detected for {symbol}")
                return None
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SCREENSHOT_DIR}/{symbol}_{timestamp}.{SCREENSHOT_FORMAT}"
            
            # 保存图片
            cv2.imwrite(filename, img)
            logger.info(f"Screenshot saved for {symbol}: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Screenshot failed for {symbol}: {e}")
            return None

    def _check_image_quality(self, img):
        """检查图像质量"""
        try:
            # 检查图像是否为空
            if img is None or img.size == 0:
                return False
                
            # 检查图像尺寸
            height, width = img.shape[:2]
            if height < 100 or width < 100:
                return False
                
            # 检查图像是否全黑或全白
            mean_value = np.mean(img)
            if mean_value < 5 or mean_value > 250:
                return False
                
            # 检查图像清晰度
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if laplacian_var < 100:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking image quality: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                logger.info("Chrome driver closed successfully")
        except Exception as e:
            logger.error(f"Failed to close Chrome driver: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import time
import os

from src.screenshot.capture import ChartCapture
from src.preprocessing.image_processor import ImageProcessor
from src.model.llava_analyzer import LLaVAAnalyzer
from src.utils.logger import setup_logger
from config.config import (
    SCREENSHOT_INTERVAL_1M,
    CHART_SYMBOLS,
    CHART_TIMEFRAME
)

logger = setup_logger('scheduler')

class TaskScheduler:
    def __init__(self):
        """初始化任务调度器"""
        self.scheduler = BackgroundScheduler()
        self.chart_capture = None
        self.image_processor = ImageProcessor()
        self.llava_analyzer = None
        self.last_analysis = {}  # 每个品种的最后分析结果
        
    def start(self):
        """启动调度器"""
        try:
            # 初始化资源
            self.chart_capture = ChartCapture()
            self.llava_analyzer = LLaVAAnalyzer()
            
            # 添加定时任务
            self.scheduler.add_job(
                self._process_1m,
                IntervalTrigger(minutes=1),
                id='1m_job',
                name='1m Chart Analysis'
            )
            
            # 启动调度器
            self.scheduler.start()
            logger.info(f"Task scheduler started for symbols: {CHART_SYMBOLS}")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.cleanup()
            raise
    
    def _process_1m(self):
        """处理1分钟任务"""
        try:
            # 获取所有品种的截图
            screenshots = self.chart_capture.capture_chart()
            
            for symbol, screenshot_path in screenshots.items():
                if not screenshot_path:
                    continue
                    
                # 检查图像质量
                if not self.image_processor.check_image_quality(screenshot_path):
                    logger.warning(f"Low quality image detected for {symbol}")
                    continue
                
                # 预处理图像
                processed_path = self.image_processor.preprocess_image(screenshot_path)
                if not processed_path:
                    continue
                
                # 裁剪图表区域
                cropped_path = self.image_processor.crop_chart_area(processed_path)
                if not cropped_path:
                    continue
                
                # LLaVA分析
                analysis_result = self.llava_analyzer.analyze_chart(cropped_path)
                
                # 检查是否需要发送信号
                if self._should_send_signal(symbol, analysis_result):
                    self._send_signal(symbol, analysis_result)
                
                # 更新最后分析结果
                self.last_analysis[symbol] = analysis_result
                
                # 清理临时文件
                self._cleanup_temp_files(processed_path, cropped_path)
                
        except Exception as e:
            logger.error(f"Error processing 1m interval: {e}")
    
    def _should_send_signal(self, symbol: str, current_result: dict) -> bool:
        """判断是否需要发送信号"""
        if symbol not in self.last_analysis:
            return True
            
        last_result = self.last_analysis[symbol]
        
        # 检查是否有新的交易信号
        if (current_result["trade_signal"]["type"] != "none" and
            current_result["trade_signal"]["type"] != last_result["trade_signal"]["type"]):
            return True
            
        # 检查SCE模式
        if (current_result["sce_pattern"]["detected"] and
            not last_result["sce_pattern"]["detected"]):
            return True
            
        # 检查双顶/双底形态
        if (current_result["double_pattern"]["confirmation"] and
            not last_result["double_pattern"]["confirmation"]):
            return True
            
        return False
    
    def _send_signal(self, symbol: str, analysis_result: dict):
        """发送交易信号"""
        try:
            signal_type = analysis_result["trade_signal"]["type"]
            
            # 构建信号消息
            message = f"交易信号 - {symbol} ({CHART_TIMEFRAME}分钟)\n"
            message += f"信号类型: {signal_type}\n"
            
            if analysis_result["trade_signal"]["entry_price"]:
                message += f"入场价格: {analysis_result['trade_signal']['entry_price']}\n"
            if analysis_result["trade_signal"]["stop_loss"]:
                message += f"止损价格: {analysis_result['trade_signal']['stop_loss']}\n"
            if analysis_result["trade_signal"]["take_profit"]:
                message += f"目标价格: {analysis_result['trade_signal']['take_profit']}\n"
                
            message += f"信号可信度: {analysis_result['trade_signal']['confidence']:.2%}\n"
            
            # 添加均线分析
            message += f"\n均线分析:\n"
            message += f"趋势: {analysis_result['market_analysis']['trend']}\n"
            if analysis_result['market_analysis']['ma_position']:
                message += f"均线位置: {analysis_result['market_analysis']['ma_position']}\n"
            
            logger.info(message)
            
        except Exception as e:
            logger.error(f"Error sending signal for {symbol}: {e}")
    
    def _cleanup_temp_files(self, *file_paths):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止调度器
            if self.scheduler.running:
                self.scheduler.shutdown()
                logger.info("Scheduler shutdown completed")
            
            # 清理其他资源
            if self.chart_capture:
                self.chart_capture.cleanup()
            
            if self.llava_analyzer:
                self.llava_analyzer.cleanup()
                
            logger.info("All resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 
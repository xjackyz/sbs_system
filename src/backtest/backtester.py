"""
回测模块，用于执行策略回测和性能分析
"""
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

from config.config import (
    CHART_SYMBOLS,
    MA_SETTINGS,
    SCREENSHOT_DIR
)
from src.model.llava_analyzer import LLaVAAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger('backtester')

@dataclass
class BacktestConfig:
    """回测配置类"""
    start_date: str  # 回测开始日期 (YYYY-MM-DD)
    end_date: str  # 回测结束日期 (YYYY-MM-DD)
    symbols: Optional[List[str]] = None  # 要回测的交易品种列表
    timeframes: Optional[List[str]] = None  # 要回测的时间周期列表
    initial_capital: float = 100000.0  # 初始资金
    position_size: float = 0.1  # 每笔交易的资金比例
    max_positions: int = 1  # 最大同时持仓数
    stop_loss: float = 0.02  # 止损比例
    take_profit: float = 0.06  # 止盈比例
    use_trailing_stop: bool = True  # 是否使用追踪止损
    trailing_stop_distance: float = 0.01  # 追踪止损距离
    commission_rate: float = 0.001  # 手续费率
    slippage: float = 0.001  # 滑点
    enable_logging: bool = True  # 是否启用日志
    save_signals: bool = True  # 是否保存信号图表
    save_trades: bool = True  # 是否保存交易记录
    risk_free_rate: float = 0.02  # 无风险利率
    validation_window: int = 100  # 验证窗口大小
    min_confidence: float = 0.75  # 最小置信度
    min_volume: float = 1000.0  # 最小成交量
    max_drawdown: float = 0.2  # 最大回撤限制
    profit_target: float = 0.5  # 利润目标
    max_trades_per_day: int = 5  # 每日最大交易次数
    min_trade_interval: int = 30  # 最小交易间隔（分钟）
    patterns: Dict[str, Dict] = None  # 模式识别配置
    filters: Dict[str, Dict] = None  # 过滤器配置
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.patterns is None:
            self.patterns = {
                'trend_continuation': {
                    'min_length': 5,
                    'max_length': 20,
                    'min_slope': 0.001,
                    'min_r2': 0.7
                },
                'trend_reversal': {
                    'min_length': 3,
                    'max_length': 10,
                    'min_angle': 30,
                    'min_volume_increase': 1.5
                }
            }
            
        if self.filters is None:
            self.filters = {
                'time': {
                    'start': '09:30',
                    'end': '16:00',
                    'exclude_holidays': True
                },
                'volume': {
                    'min_threshold': self.min_volume,
                    'increase_factor': 1.5
                },
                'volatility': {
                    'min': 0.001,
                    'max': 0.05
                },
                'trend': {
                    'lookback': 20,
                    'min_strength': 0.6
                }
            }

class Backtester:
    def __init__(self, config: BacktestConfig):
        """
        初始化回测系统
        
        Args:
            config: 回测配置
        """
        try:
            logger.info(f"Initializing backtester for period: {config.start_date} to {config.end_date}")
            self.config = config
            self.start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
            self.end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
            self.data = {}  # 存储历史数据
            self.signals = {}  # 存储生成的信号
            self.trades = {}  # 存储交易记录
            self.equity = config.initial_capital  # 当前资金
            self.max_drawdown = 0.0  # 最大回撤
            self.peak_equity = config.initial_capital  # 峰值资金
            
            # 创建信号截图目录
            self.signal_dir = os.path.join(SCREENSHOT_DIR, "backtest_signals")
            if not os.path.exists(self.signal_dir):
                os.makedirs(self.signal_dir)
                logger.info(f"Created signal directory: {self.signal_dir}")
            
            # 初始化LLaVA分析器
            logger.info("Initializing LLaVA analyzer...")
            self.llava_analyzer = LLaVAAnalyzer()
            logger.info("LLaVA analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backtester: {e}", exc_info=True)
            raise
    
    def load_data(self):
        """加载历史数据"""
        try:
            logger.info("Loading historical data...")
            for symbol in CHART_SYMBOLS:
                try:
                    # 从CSV文件加载数据
                    filename = f"data/{symbol}_{self.start_date.strftime('%Y%m')}_1m.csv"
                    logger.info(f"Loading data from: {filename}")
                    
                    if not os.path.exists(filename):
                        logger.error(f"Data file not found: {filename}")
                        raise FileNotFoundError(f"Data file not found: {filename}")
                    
                    df = pd.read_csv(filename)
                    logger.info(f"Loaded {len(df)} rows of data")
                    
                    # 检查必要的列是否存在
                    required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        logger.error(f"Missing required columns: {missing_columns}")
                        raise ValueError(f"Missing required columns: {missing_columns}")
                    
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    # 检查数据范围
                    logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
                    
                    # 计算技术指标
                    logger.info("Calculating technical indicators...")
                    self._calculate_indicators(df)
                    
                    self.data[symbol] = df
                    logger.info(f"Data loaded successfully for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}", exc_info=True)
                    raise
                
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def _calculate_indicators(self, df: pd.DataFrame):
        """计算技术指标"""
        try:
            logger.info("Calculating SMA indicators...")
            # 计算SMA
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA200'] = df['close'].rolling(window=200).mean()
            
            logger.info("Calculating candle colors...")
            # 计算蜡烛颜色（用于SCE模式识别）
            df['candle_color'] = np.where(df['close'] >= df['open'], 1, -1)
            
            logger.info("Calculating rolling high/low...")
            # 计算高低点范围（用于双顶/双底识别）
            df['high_roll'] = df['high'].rolling(window=20).max()
            df['low_roll'] = df['low'].rolling(window=20).min()
            
            logger.info("Technical indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            raise
    
    def run_backtest(self):
        """运行回测"""
        try:
            logger.info("Starting backtest...")
            
            # 加载数据
            logger.info("Loading historical data...")
            self.load_data()
            logger.info("Historical data loaded successfully")
            
            # 对每个品种进行回测
            for symbol in CHART_SYMBOLS:
                logger.info(f"Starting backtest for {symbol}")
                df = self.data[symbol]
                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Data columns: {df.columns.tolist()}")
                logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
                
                # 初始化信号和交易列表
                self.signals[symbol] = []
                self.trades[symbol] = []
                
                # 遍历每个时间点
                total_bars = len(df)
                processed_bars = 0
                
                for timestamp in df.index:
                    if timestamp < self.start_date or timestamp > self.end_date:
                        continue
                    
                    processed_bars += 1
                    if processed_bars % 100 == 0:
                        logger.info(f"Progress: {processed_bars}/{total_bars} bars processed")
                    
                    try:
                        # 生成图表图像
                        logger.info(f"Generating chart for {symbol} at {timestamp}")
                        chart_image = self.generate_chart_image(symbol, timestamp)
                        if not chart_image:
                            logger.warning(f"Failed to generate chart for {symbol} at {timestamp}")
                            continue
                        
                        # 使用LLaVA分析器分析图表
                        logger.info(f"Analyzing chart: {chart_image}")
                        analysis_result = self.llava_analyzer.analyze_chart(chart_image)
                        logger.info(f"Analysis result: {analysis_result}")
                        
                        # 如果发现交易信号，保存图表和信号信息
                        if self._is_valid_signal(analysis_result):
                            logger.info(f"Valid signal detected at {timestamp}")
                            signal_info = self._process_signal(symbol, timestamp, analysis_result)
                            self.signals[symbol].append(signal_info)
                            logger.info(f"Signal processed: {signal_info}")
                            
                            # 保存信号图表（用于半监督学习）
                            self._save_signal_chart(chart_image, signal_info)
                            logger.info(f"Signal chart saved for {symbol} at {timestamp}")
                        else:
                            logger.info(f"No valid signal detected at {timestamp}")
                        
                        # 清理临时图表文件
                        if os.path.exists(chart_image):
                            os.remove(chart_image)
                            logger.info(f"Temporary chart file removed: {chart_image}")
                            
                    except Exception as e:
                        logger.error(f"Error processing bar at {timestamp}: {e}", exc_info=True)
                        continue
                
                logger.info(f"Completed backtest for {symbol}")
                logger.info(f"Total signals detected: {len(self.signals[symbol])}")
            
            # 生成回测报告
            logger.info("Generating backtest report...")
            self._generate_report()
            logger.info("Backtest report generated successfully")
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}", exc_info=True)
            raise
        finally:
            logger.info("Starting cleanup...")
            self.cleanup()
            logger.info("Cleanup completed")
    
    def generate_chart_image(self, symbol: str, timestamp: datetime) -> str:
        """生成特定时间点的图表图像"""
        try:
            logger.info(f"Generating chart image for {symbol} at {timestamp}")
            
            # 获取历史数据切片（前200根K线）
            df = self.data[symbol]
            end_idx = df.index.get_loc(timestamp)
            start_idx = max(0, end_idx - 200)
            chart_data = df.iloc[start_idx:end_idx + 1]
            
            logger.info(f"Got {len(chart_data)} bars of historical data")
            
            # 创建图表图像
            fig_path = self._create_chart_image(chart_data, symbol, timestamp)
            if fig_path:
                logger.info(f"Chart image saved to: {fig_path}")
            return fig_path
            
        except Exception as e:
            logger.error(f"Error generating chart image: {e}", exc_info=True)
            return None
    
    def _create_chart_image(self, data: pd.DataFrame, symbol: str, timestamp: datetime) -> str:
        """创建图表图像"""
        try:
            logger.info("Creating chart image...")
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制K线图
            ohlc = data[['open', 'high', 'low', 'close']].copy()
            ohlc.index = mdates.date2num(ohlc.index.to_pydatetime())
            candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')
            
            # 添加均线
            ax.plot(ohlc.index, data['SMA20'], 'b-', alpha=0.5, label='SMA20')
            ax.plot(ohlc.index, data['SMA200'], 'purple', alpha=0.5, label='SMA200')
            
            # 设置图表格式
            ax.grid(True)
            ax.legend()
            
            # 保存图表
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(self.signal_dir, f"{symbol}_{timestamp_str}.png")
            plt.savefig(fig_path)
            plt.close()
            
            logger.info(f"Chart image created: {fig_path}")
            return fig_path
            
        except Exception as e:
            logger.error(f"Error creating chart image: {e}", exc_info=True)
            return None
    
    def _is_valid_signal(self, analysis_result: Dict) -> bool:
        """检查是否为有效的SBS序列信号"""
        try:
            # 检查序列是否存在
            if not analysis_result.get("sequence", {}).get("detected", False):
                return False
                
            # 检查序列状态
            sequence_status = analysis_result.get("sequence", {}).get("status", "invalid")
            if sequence_status == "invalid":
                return False
                
            # 检查必要的点位是否都已形成
            points = analysis_result.get("sequence", {}).get("points", {})
            required_points = ["breakout", "point1", "point2", "point3", "point4"]
            if not all(point in points for point in required_points):
                return False
                
            # 检查序列有效性条件
            validity = analysis_result.get("sequence", {}).get("validity", {})
            if validity.get("point3_break", False):  # 点3超过突破点
                return False
            if validity.get("structure_break", False):  # 结构破坏
                return False
            if not validity.get("volume_support", False):  # 成交量不支持
                return False
                
            # 检查交易信号
            trade_signal = analysis_result.get("trade_signal", {})
            if not trade_signal.get("entry_price"):
                return False
            if not trade_signal.get("stop_loss"):
                return False
            if not trade_signal.get("take_profit"):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal validity: {e}", exc_info=True)
            return False
    
    def _process_signal(self, symbol: str, timestamp: datetime, analysis_result: Dict) -> Dict:
        """处理SBS序列交易信号"""
        try:
            sequence = analysis_result.get("sequence", {})
            trade_signal = analysis_result.get("trade_signal", {})
            
            signal_info = {
                "symbol": symbol,
                "timestamp": timestamp,
                "sequence_type": sequence.get("type", "unknown"),  # 上升序列或下降序列
                "points": {
                    "breakout": sequence.get("points", {}).get("breakout"),
                    "point1": sequence.get("points", {}).get("point1"),
                    "point2": sequence.get("points", {}).get("point2"),
                    "point3": sequence.get("points", {}).get("point3"),
                    "point4": sequence.get("points", {}).get("point4"),
                    "point5": sequence.get("points", {}).get("point5")
                },
                "entry_price": trade_signal.get("entry_price"),
                "stop_loss": trade_signal.get("stop_loss"),
                "take_profit": trade_signal.get("take_profit"),
                "confidence": trade_signal.get("confidence", 0.0),
                "volume_support": sequence.get("validity", {}).get("volume_support", False)
            }
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}", exc_info=True)
            return {
                "symbol": symbol,
                "timestamp": timestamp,
                "error": str(e)
            }
    
    def _save_signal_chart(self, chart_image: str, signal_info: Dict):
        """保存SBS序列信号图表（用于半监督学习）"""
        try:
            # 创建信号类型目录
            signal_type_dir = os.path.join(self.signal_dir, signal_info["sequence_type"])
            if not os.path.exists(signal_type_dir):
                os.makedirs(signal_type_dir)
            
            # 构建文件名
            timestamp_str = signal_info["timestamp"].strftime("%Y%m%d_%H%M%S")
            filename = f"{signal_info['symbol']}_{timestamp_str}.png"
            
            # 保存图表
            save_path = os.path.join(signal_type_dir, filename)
            
            # 添加信号信息到图表
            img = cv2.imread(chart_image)
            self._add_signal_info_to_image(img, signal_info)
            cv2.imwrite(save_path, img)
            
            # 保存序列点位信息
            info_path = save_path.replace('.png', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Symbol: {signal_info['symbol']}\n")
                f.write(f"Timestamp: {signal_info['timestamp']}\n")
                f.write(f"Sequence Type: {signal_info['sequence_type']}\n")
                f.write("\nSequence Points:\n")
                for point_name, point_value in signal_info['points'].items():
                    f.write(f"{point_name}: {point_value}\n")
                f.write(f"\nEntry Price: {signal_info['entry_price']}\n")
                f.write(f"Stop Loss: {signal_info['stop_loss']}\n")
                f.write(f"Take Profit: {signal_info['take_profit']}\n")
                f.write(f"Confidence: {signal_info['confidence']:.2%}\n")
                f.write(f"Volume Support: {signal_info['volume_support']}\n")
            
            logger.info(f"Signal chart and info saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving signal chart: {e}", exc_info=True)
    
    def _add_signal_info_to_image(self, img: np.ndarray, signal_info: Dict):
        """在图表上添加SBS序列信号信息"""
        try:
            # 添加文本信息
            text_lines = [
                f"Symbol: {signal_info['symbol']}",
                f"Type: {signal_info['sequence_type']}",
                f"Entry: {signal_info['entry_price']}",
                f"SL: {signal_info['stop_loss']}",
                f"TP: {signal_info['take_profit']}",
                f"Conf: {signal_info['confidence']:.2%}"
            ]
            
            # 添加序列点位信息
            for point_name, point_value in signal_info['points'].items():
                if point_value:
                    text_lines.append(f"{point_name}: {point_value}")
            
            y = 30
            for line in text_lines:
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 2)
                y += 30
                
        except Exception as e:
            logger.error(f"Error adding signal info to image: {e}", exc_info=True)
    
    def _generate_report(self):
        """生成回测报告"""
        try:
            report = {
                "period": f"{self.start_date.date()} to {self.end_date.date()}",
                "signals": {
                    symbol: len(signals) for symbol, signals in self.signals.items()
                },
                "sequence_types": self._analyze_sequence_types(),
                "success_rate": self._calculate_success_rate(),
                "average_profit": self._calculate_average_profit(),
                "time_distribution": self._analyze_time_distribution()
            }
            
            # 保存报告
            report_path = os.path.join(self.signal_dir, "backtest_report.txt")
            with open(report_path, 'w') as f:
                f.write("SBS序列回测报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"回测周期: {report['period']}\n\n")
                
                f.write("信号统计:\n")
                for symbol, count in report['signals'].items():
                    f.write(f"{symbol}: {count}个信号\n")
                f.write("\n")
                
                f.write("序列类型分布:\n")
                for seq_type, count in report['sequence_types'].items():
                    f.write(f"{seq_type}: {count}个\n")
                f.write("\n")
                
                f.write(f"成功率: {report['success_rate']:.2%}\n")
                f.write(f"平均盈利: {report['average_profit']:.2f}\n\n")
                
                f.write("时间分布:\n")
                for hour, count in report['time_distribution'].items():
                    f.write(f"{hour}时: {count}个信号\n")
            
            logger.info("Backtest report generated")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
    
    def _analyze_sequence_types(self) -> Dict:
        """分析序列类型分布"""
        sequence_types = {}
        for symbol_signals in self.signals.values():
            for signal in symbol_signals:
                seq_type = signal["sequence_type"]
                sequence_types[seq_type] = sequence_types.get(seq_type, 0) + 1
        return sequence_types
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        try:
            total_signals = 0
            successful_signals = 0
            
            for symbol_signals in self.signals.values():
                for signal in symbol_signals:
                    total_signals += 1
                    # 这里需要实现具体的成功判定逻辑
                    # 例如：是否达到目标价位，或是否在止损前达到一定盈利
            
            return successful_signals / total_signals if total_signals > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}", exc_info=True)
            return 0.0
    
    def _calculate_average_profit(self) -> float:
        """计算平均盈利"""
        try:
            total_profit = 0.0
            total_trades = 0
            
            for symbol_signals in self.signals.values():
                for signal in symbol_signals:
                    total_trades += 1
                    # 这里需要实现具体的盈利计算逻辑
                    # 例如：根据入场价格、止损和目标价位计算理论盈利
            
            return total_profit / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average profit: {e}", exc_info=True)
            return 0.0
    
    def _analyze_time_distribution(self) -> Dict:
        """分析信号时间分布"""
        time_dist = {}
        for symbol_signals in self.signals.values():
            for signal in symbol_signals:
                hour = signal["timestamp"].hour
                time_dist[hour] = time_dist.get(hour, 0) + 1
        return dict(sorted(time_dist.items()))
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("Cleaning up resources...")
            if self.llava_analyzer:
                self.llava_analyzer.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True) 
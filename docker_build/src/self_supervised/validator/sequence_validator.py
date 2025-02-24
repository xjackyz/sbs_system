import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import ta

from config.config import EVALUATION_SYSTEM
from src.utils.logger import setup_logger

logger = setup_logger('sequence_validator')

@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    score: float
    details: Dict
    stage: str
    timestamp: str = datetime.now().isoformat()

class SequenceValidator:
    """序列验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.rules = EVALUATION_SYSTEM['sequence_validation']
        self.scoring = EVALUATION_SYSTEM['scoring_weights']
        self.thresholds = EVALUATION_SYSTEM['thresholds']
        self.sequence_types = EVALUATION_SYSTEM['sequence_types']
        
    def validate_sequence(self, sequence_data: pd.DataFrame, 
                         prediction: Dict) -> ValidationResult:
        """
        验证序列
        
        Args:
            sequence_data: 序列数据
            prediction: 预测结果
            
        Returns:
            ValidationResult: 验证结果
        """
        try:
            # 1. 突破阶段验证
            breakout_result = self._validate_breakout(sequence_data, prediction)
            if not breakout_result['is_valid']:
                return ValidationResult(
                    is_valid=False,
                    score=breakout_result['score'],
                    details={'breakout': breakout_result},
                    stage='breakout'
                )
            
            # 2. 回调阶段验证
            pullback_result = self._validate_pullback(sequence_data, prediction)
            if not pullback_result['is_valid']:
                return ValidationResult(
                    is_valid=False,
                    score=(breakout_result['score'] + pullback_result['score']) / 2,
                    details={
                        'breakout': breakout_result,
                        'pullback': pullback_result
                    },
                    stage='pullback'
                )
            
            # 3. 确认阶段验证
            confirmation_result = self._validate_confirmation(sequence_data, prediction)
            
            # 计算总分
            total_score = (
                breakout_result['score'] * self.rules['breakout']['score_weight'] +
                pullback_result['score'] * self.rules['pullback']['score_weight'] +
                confirmation_result['score'] * self.rules['confirmation']['score_weight']
            )
            
            # 检查是否达到最小分数要求
            is_valid = total_score >= self.thresholds['min_score']
            
            return ValidationResult(
                is_valid=is_valid,
                score=total_score,
                details={
                    'breakout': breakout_result,
                    'pullback': pullback_result,
                    'confirmation': confirmation_result
                },
                stage='complete'
            )
            
        except Exception as e:
            logger.error(f"序列验证失败: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                details={'error': str(e)},
                stage='error'
            )
            
    def _validate_breakout(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """验证突破阶段"""
        try:
            # 计算市场结构
            market_structure = self._analyze_market_structure(data)
            
            # 计算趋势强度
            trend_strength = self._calculate_trend_strength(data)
            
            # 计算波动性
            volatility = self._calculate_volatility(data)
            
            # 验证突破
            price_break = self._check_price_break(data, prediction['breakout_index'], prediction['direction'])
            pattern_valid = self._check_candle_pattern(data, prediction['breakout_index'], prediction['direction'])
            
            # 计算综合得分
            score = (
                price_break * 0.6 +
                pattern_valid * 0.4
            ) * (
                market_structure['score'] * 0.3 +
                trend_strength['score'] * 0.4 +
                volatility['score'] * 0.3
            )
            
            return {
                'is_valid': score >= self.thresholds['min_score'],
                'score': score,
                'details': {
                    'price_break': price_break,
                    'pattern_valid': pattern_valid,
                    'market_structure': market_structure,
                    'trend_strength': trend_strength,
                    'volatility': volatility
                }
            }
            
        except Exception as e:
            logger.error(f"突破阶段验证失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'details': {}}
            
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """分析市场结构"""
        try:
            # 计算支撑和阻力位
            pivots = self._find_pivot_points(data)
            
            # 计算趋势线
            trend_lines = self._calculate_trend_lines(data)
            
            # 计算市场结构得分
            structure_score = (
                pivots['strength'] * 0.5 +
                trend_lines['strength'] * 0.5
            )
            
            return {
                'score': structure_score,
                'pivots': pivots,
                'trend_lines': trend_lines
            }
            
        except Exception as e:
            logger.error(f"市场结构分析失败: {e}")
            return {'score': 0.0}
            
    def _find_pivot_points(self, data: pd.DataFrame) -> Dict:
        """查找关键支撑阻力位"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            window = 20
            
            # 查找局部高点和低点
            pivot_highs = []
            pivot_lows = []
            
            for i in range(window, len(data) - window):
                # 高点
                if highs[i] == max(highs[i-window:i+window+1]):
                    pivot_highs.append((i, highs[i]))
                # 低点
                if lows[i] == min(lows[i-window:i+window+1]):
                    pivot_lows.append((i, lows[i]))
            
            # 计算支撑阻力强度
            strength = len(pivot_highs) + len(pivot_lows)
            normalized_strength = min(strength / 10, 1.0)  # 最多10个关键点
            
            return {
                'strength': normalized_strength,
                'highs': pivot_highs,
                'lows': pivot_lows
            }
            
        except Exception as e:
            logger.error(f"支撑阻力位分析失败: {e}")
            return {'strength': 0.0}
            
    def _calculate_trend_lines(self, data: pd.DataFrame) -> Dict:
        """计算趋势线"""
        try:
            prices = data['close'].values
            x = np.arange(len(prices))
            
            # 线性回归
            slope, intercept = np.polyfit(x, prices, 1)
            
            # 计算R方值
            y_pred = slope * x + intercept
            r2 = 1 - np.sum((prices - y_pred) ** 2) / np.sum((prices - prices.mean()) ** 2)
            
            # 计算趋势强度
            trend_strength = min(abs(slope) * 1000, 1.0)  # 归一化斜率
            
            return {
                'strength': trend_strength,
                'slope': slope,
                'intercept': intercept,
                'r2': r2
            }
            
        except Exception as e:
            logger.error(f"趋势线计算失败: {e}")
            return {'strength': 0.0}
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """计算趋势强度"""
        try:
            # 计算移动平均
            ma20 = data['close'].rolling(window=20).mean()
            ma50 = data['close'].rolling(window=50).mean()
            
            # 计算趋势方向
            trend_direction = 1 if ma20.iloc[-1] > ma50.iloc[-1] else -1
            
            # 计算价格与均线的距离
            price_ma_distance = abs(data['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
            
            # 计算动量指标
            rsi = ta.momentum.rsi(data['close'])
            macd = ta.trend.macd_diff(data['close'])
            
            # 计算趋势强度得分
            trend_score = (
                (price_ma_distance * 0.3) +
                (abs(rsi.iloc[-1] - 50) / 50 * 0.3) +
                (abs(macd.iloc[-1]) / data['close'].iloc[-1] * 0.4)
            )
            
            return {
                'score': min(trend_score, 1.0),
                'direction': trend_direction,
                'details': {
                    'price_ma_distance': price_ma_distance,
                    'rsi': rsi.iloc[-1],
                    'macd': macd.iloc[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"趋势强度计算失败: {e}")
            return {'score': 0.0}
            
    def _calculate_volatility(self, data: pd.DataFrame) -> Dict:
        """计算波动性"""
        try:
            # 计算真实波幅
            atr = ta.volatility.average_true_range(
                data['high'],
                data['low'],
                data['close']
            )
            
            # 计算布林带
            bb = ta.volatility.BollingerBands(
                data['close'],
                window=20,
                window_dev=2
            )
            
            # 计算波动率
            volatility = (data['high'] - data['low']) / data['close']
            
            # 计算波动性得分
            volatility_score = (
                (atr.iloc[-1] / data['close'].iloc[-1] * 0.4) +
                (bb.bandwidth.iloc[-1] * 0.3) +
                (volatility.mean() * 0.3)
            )
            
            return {
                'score': min(volatility_score * 10, 1.0),  # 归一化
                'details': {
                    'atr': atr.iloc[-1],
                    'bb_bandwidth': bb.bandwidth.iloc[-1],
                    'avg_volatility': volatility.mean()
                }
            }
            
        except Exception as e:
            logger.error(f"波动性计算失败: {e}")
            return {'score': 0.0}
            
    def _validate_pullback(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """验证回调阶段"""
        try:
            rules = self.rules['pullback']
            result = {
                'is_valid': False,
                'score': 0.0,
                'details': {}
            }
            
            # 1. 检查回调水平
            retracement = self._check_retracement_level(
                data,
                prediction['point1_index'],
                prediction['point2_index']
            )
            result['details']['retracement'] = retracement
            
            # 2. 检查成交量减少
            volume_decrease = self._check_volume_decrease(
                data,
                prediction['point1_index'],
                prediction['point2_index']
            )
            result['details']['volume_decrease'] = volume_decrease
            
            # 3. 检查支撑/阻力位
            support_resistance = self._check_support_resistance(
                data,
                prediction['point1_index'],
                prediction['point2_index']
            )
            result['details']['support_resistance'] = support_resistance
            
            # 计算得分
            score = (
                retracement * rules['retracement']['min_level'] +
                volume_decrease * rules['retracement']['volume_decrease'] +
                support_resistance * rules['support_resistance']['touch_count']
            )
            
            result['score'] = score
            result['is_valid'] = score >= self.thresholds['min_score']
            
            return result
            
        except Exception as e:
            logger.error(f"回调阶段验证失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'details': {'error': str(e)}}
            
    def _validate_confirmation(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """验证确认阶段"""
        try:
            rules = self.rules['confirmation']
            result = {
                'is_valid': False,
                'score': 0.0,
                'details': {}
            }
            
            # 1. 检查双底/双顶形态
            double_pattern = self._check_double_pattern(
                data,
                prediction['point3_index'],
                prediction['point4_index']
            )
            result['details']['double_pattern'] = double_pattern
            
            # 2. 检查流动性获取
            liquidity = self._check_liquidation(
                data,
                prediction['point3_index'],
                prediction['point4_index']
            )
            result['details']['liquidity'] = liquidity
            
            # 3. 检查SCE信号
            sce_signal = self._check_sce_signal(
                data,
                prediction['point5_index']
            )
            result['details']['sce_signal'] = sce_signal
            
            # 计算得分
            score = (
                double_pattern * rules['double_pattern']['price_similarity'] +
                liquidity * rules['liquidity']['min_range'] +
                sce_signal * rules['sce_signal']['confidence']
            )
            
            result['score'] = score
            result['is_valid'] = score >= self.thresholds['min_score']
            
            return result
            
        except Exception as e:
            logger.error(f"确认阶段验证失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'details': {'error': str(e)}}
            
    def _check_price_break(self, data: pd.DataFrame, index: int, direction: str) -> bool:
        """检查价格突破"""
        try:
            if direction == 'up':
                # 检查是否突破前期高点
                prev_high = data['high'].iloc[max(0, index-20):index].max()
                return data['close'].iloc[index] > prev_high
            else:
                # 检查是否突破前期低点
                prev_low = data['low'].iloc[max(0, index-20):index].min()
                return data['close'].iloc[index] < prev_low
                
        except Exception as e:
            logger.error(f"价格突破检查失败: {e}")
            return False
            
    def _check_candle_pattern(self, data: pd.DataFrame, index: int, direction: str) -> bool:
        """检查K线形态"""
        try:
            if direction == 'up':
                # 检查是否为看涨K线形态
                return (data['close'].iloc[index] > data['open'].iloc[index] and
                       data['close'].iloc[index] > data['close'].iloc[index-1])
            else:
                # 检查是否为看跌K线形态
                return (data['close'].iloc[index] < data['open'].iloc[index] and
                       data['close'].iloc[index] < data['close'].iloc[index-1])
                
        except Exception as e:
            logger.error(f"K线形态检查失败: {e}")
            return False
            
    def _check_retracement_level(self, data: pd.DataFrame, point1_idx: int, point2_idx: int) -> bool:
        """检查回调水平"""
        try:
            point1_price = data.iloc[point1_idx]['close']
            point2_price = data.iloc[point2_idx]['close']
            price_range = abs(point2_price - point1_price)
            
            retracement = price_range / point1_price
            return 0.382 <= retracement <= 0.618
            
        except Exception as e:
            logger.error(f"回调水平检查失败: {e}")
            return False
            
    def _check_volume_decrease(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """检查成交量减少"""
        try:
            start_volume = data.iloc[start_idx]['volume']
            end_volume = data.iloc[end_idx]['volume']
            return end_volume < start_volume * 0.7
            
        except Exception as e:
            logger.error(f"成交量减少检查失败: {e}")
            return False
            
    def _check_support_resistance(self, data: pd.DataFrame, point1_idx: int, point2_idx: int) -> bool:
        """检查支撑/阻力位"""
        try:
            price_level = data.iloc[point1_idx]['close']
            price_tolerance = price_level * 0.001
            
            # 统计触及次数
            touches = 0
            for i in range(point1_idx, point2_idx + 1):
                if abs(data.iloc[i]['high'] - price_level) <= price_tolerance or \
                   abs(data.iloc[i]['low'] - price_level) <= price_tolerance:
                    touches += 1
                    
            return touches >= 2
            
        except Exception as e:
            logger.error(f"支撑/阻力位检查失败: {e}")
            return False
            
    def _check_double_pattern(self, data: pd.DataFrame, point3_idx: int, point4_idx: int) -> bool:
        """检查双底/双顶形态"""
        try:
            point3_price = data.iloc[point3_idx]['close']
            point4_price = data.iloc[point4_idx]['close']
            
            # 检查价格相似度
            price_diff = abs(point4_price - point3_price) / point3_price
            return price_diff <= 0.002
            
        except Exception as e:
            logger.error(f"双底/双顶形态检查失败: {e}")
            return False
            
    def _check_liquidation(self, data: pd.DataFrame, point3_idx: int, point4_idx: int) -> bool:
        """检查流动性获取"""
        try:
            # 计算区间最高价和最低价
            high_prices = data.iloc[point3_idx:point4_idx+1]['high']
            low_prices = data.iloc[point3_idx:point4_idx+1]['low']
            
            price_range = (high_prices.max() - low_prices.min()) / low_prices.min()
            return price_range >= 0.003
            
        except Exception as e:
            logger.error(f"流动性获取检查失败: {e}")
            return False
            
    def _check_sce_signal(self, data: pd.DataFrame, index: int) -> bool:
        """检查SCE信号"""
        try:
            # 这里需要根据实际的SCE信号判断逻辑来实现
            # 当前使用一个简化的判断
            candle = data.iloc[index]
            prev_candle = data.iloc[index-1]
            
            return abs(candle['close'] - prev_candle['close']) / prev_candle['close'] > 0.001
            
        except Exception as e:
            logger.error(f"SCE信号检查失败: {e}")
            return False 
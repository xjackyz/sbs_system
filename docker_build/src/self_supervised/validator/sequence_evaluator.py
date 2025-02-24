import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SBSPoint:
    """SBS序列中的关键点"""
    index: int
    price: float
    timestamp: str
    point_type: str  # breakout, point1, point2, point3, point4, point5
    confidence: float

@dataclass
class RewardConfig:
    """奖励配置"""
    breakout_reward: float = 1.0
    false_breakout_penalty: float = -1.0
    point1_reward: float = 1.0
    point2_reward: float = 1.0
    point3_reward: float = 1.0
    point4_reward: float = 1.0
    point5_reward: float = 1.0
    invalid_sequence_penalty: float = -2.0
    early_exit_penalty: float = -1.0
    successful_trade_reward: float = 2.0

class SequenceEvaluator:
    """SBS序列评估器"""
    
    def __init__(self, reward_config: Optional[RewardConfig] = None):
        """
        初始化评估器
        
        Args:
            reward_config: 奖励配置
        """
        self.reward_config = reward_config or RewardConfig()
        
    def evaluate_breakout(self, data: Dict, prediction: Dict) -> Tuple[float, Dict]:
        """
        评估突破阶段
        
        Args:
            data: 市场数据
            prediction: 模型预测结果
            
        Returns:
            reward: 奖励值
            metrics: 评估指标
        """
        reward = 0.0
        metrics = {'valid': False, 'confidence': 0.0}
        
        try:
            # 获取预测的突破点
            breakout = prediction.get('breakout', {})
            if not breakout:
                return self.reward_config.false_breakout_penalty, metrics
                
            # 验证突破
            price_data = data['price_data']
            breakout_index = breakout.get('index', 0)
            breakout_price = breakout.get('price', 0.0)
            
            # 检查是否是有效突破
            is_valid = self._validate_breakout(
                price_data,
                breakout_index,
                breakout_price,
                breakout.get('direction', 'up')
            )
            
            if is_valid:
                reward = self.reward_config.breakout_reward
                metrics['valid'] = True
                metrics['confidence'] = breakout.get('confidence', 0.0)
            else:
                reward = self.reward_config.false_breakout_penalty
                
        except Exception as e:
            logger.error(f"突破评估失败: {e}")
            reward = self.reward_config.false_breakout_penalty
            
        return reward, metrics
        
    def evaluate_retracement(self, data: Dict, prediction: Dict) -> Tuple[float, Dict]:
        """
        评估回调阶段
        
        Args:
            data: 市场数据
            prediction: 模型预测结果
            
        Returns:
            reward: 奖励值
            metrics: 评估指标
        """
        reward = 0.0
        metrics = {'valid': False, 'confidence': 0.0}
        
        try:
            # 获取预测的点1
            point1 = prediction.get('point1', {})
            if not point1:
                return 0.0, metrics
                
            # 验证回调
            price_data = data['price_data']
            point1_index = point1.get('index', 0)
            point1_price = point1.get('price', 0.0)
            
            # 检查回调的有效性
            is_valid = self._validate_retracement(
                price_data,
                point1_index,
                point1_price,
                prediction.get('breakout', {})
            )
            
            if is_valid:
                reward = self.reward_config.point1_reward
                metrics['valid'] = True
                metrics['confidence'] = point1.get('confidence', 0.0)
                
        except Exception as e:
            logger.error(f"回调评估失败: {e}")
            
        return reward, metrics
        
    def evaluate_confirmation(self, data: Dict, prediction: Dict) -> Tuple[float, Dict]:
        """
        评估确认阶段
        
        Args:
            data: 市场数据
            prediction: 模型预测结果
            
        Returns:
            reward: 奖励值
            metrics: 评估指标
        """
        reward = 0.0
        metrics = {
            'valid': False,
            'confidence': 0.0,
            'pattern_type': None
        }
        
        try:
            # 获取点3和点4
            point3 = prediction.get('point3', {})
            point4 = prediction.get('point4', {})
            
            if not point3 or not point4:
                return 0.0, metrics
                
            # 验证双底/双顶形态
            is_valid_pattern = self._validate_double_pattern(
                data['price_data'],
                point3.get('index', 0),
                point4.get('index', 0),
                prediction.get('pattern_type', '')
            )
            
            # 验证流动性获取
            is_valid_liquidity = self._validate_liquidity_grab(
                data['price_data'],
                point3.get('index', 0),
                point3.get('price', 0.0),
                prediction.get('direction', 'up')
            )
            
            if is_valid_pattern and is_valid_liquidity:
                reward = self.reward_config.point3_reward + self.reward_config.point4_reward
                metrics['valid'] = True
                metrics['confidence'] = (point3.get('confidence', 0.0) + point4.get('confidence', 0.0)) / 2
                metrics['pattern_type'] = prediction.get('pattern_type', '')
                
        except Exception as e:
            logger.error(f"确认阶段评估失败: {e}")
            
        return reward, metrics
        
    def evaluate_trend_continuation(self, data: Dict, prediction: Dict) -> Tuple[float, Dict]:
        """
        评估趋势延续阶段
        
        Args:
            data: 市场数据
            prediction: 模型预测结果
            
        Returns:
            reward: 奖励值
            metrics: 评估指标
        """
        reward = 0.0
        metrics = {'valid': False, 'profit': 0.0}
        
        try:
            # 获取点5和止盈止损位
            point5 = prediction.get('point5', {})
            take_profit = prediction.get('take_profit', 0.0)
            stop_loss = prediction.get('stop_loss', 0.0)
            
            if not point5 or not take_profit or not stop_loss:
                return 0.0, metrics
                
            # 计算实际盈亏
            entry_price = point4.get('price', 0.0)
            exit_price = point5.get('price', 0.0)
            direction = prediction.get('direction', 'up')
            
            profit = self._calculate_profit(
                entry_price,
                exit_price,
                direction,
                take_profit,
                stop_loss
            )
            
            metrics['profit'] = profit
            
            if profit > 0:
                reward = self.reward_config.successful_trade_reward
                metrics['valid'] = True
            elif profit < 0:
                reward = self.reward_config.early_exit_penalty
                
        except Exception as e:
            logger.error(f"趋势延续评估失败: {e}")
            
        return reward, metrics
        
    def _validate_breakout(self, price_data: np.ndarray, index: int, price: float, direction: str) -> bool:
        """验证突破的有效性"""
        try:
            if index < 20:  # 需要足够的历史数据
                return False
                
            # 获取之前的价格范围
            prev_prices = price_data[index-20:index]
            prev_high = np.max(prev_prices)
            prev_low = np.min(prev_prices)
            
            if direction == 'up':
                return price > prev_high
            else:
                return price < prev_low
                
        except Exception as e:
            logger.error(f"突破验证失败: {e}")
            return False
            
    def _validate_retracement(self, price_data: np.ndarray, index: int, price: float, breakout: Dict) -> bool:
        """验证回调的有效性"""
        try:
            if not breakout or index <= breakout.get('index', 0):
                return False
                
            # 计算回调幅度
            breakout_price = breakout.get('price', 0.0)
            retracement = abs(price - breakout_price) / abs(breakout_price)
            
            # 回调幅度应在30%-70%之间
            return 0.3 <= retracement <= 0.7
            
        except Exception as e:
            logger.error(f"回调验证失败: {e}")
            return False
            
    def _validate_double_pattern(self, price_data: np.ndarray, point3_index: int, point4_index: int, pattern_type: str) -> bool:
        """验证双底/双顶形态"""
        try:
            if point4_index <= point3_index:
                return False
                
            # 获取两个点位的价格
            point3_price = price_data[point3_index]
            point4_price = price_data[point4_index]
            
            # 价格相似度检查
            price_diff = abs(point4_price - point3_price) / point3_price
            
            # 时间间隔检查
            time_diff = point4_index - point3_index
            
            return price_diff <= 0.01 and 5 <= time_diff <= 20
            
        except Exception as e:
            logger.error(f"双底/双顶验证失败: {e}")
            return False
            
    def _validate_liquidity_grab(self, price_data: np.ndarray, index: int, price: float, direction: str) -> bool:
        """验证流动性获取"""
        try:
            if index < 5:
                return False
                
            # 检查是否突破了前期高点/低点
            prev_prices = price_data[index-5:index]
            
            if direction == 'up':
                return price > np.max(prev_prices)
            else:
                return price < np.min(prev_prices)
                
        except Exception as e:
            logger.error(f"流动性获取验证失败: {e}")
            return False
            
    def _calculate_profit(self, entry_price: float, exit_price: float, direction: str,
                         take_profit: float, stop_loss: float) -> float:
        """计算交易盈亏"""
        try:
            if direction == 'up':
                profit = (exit_price - entry_price) / entry_price
            else:
                profit = (entry_price - exit_price) / entry_price
                
            # 检查是否触及止盈止损
            if direction == 'up':
                if exit_price >= take_profit:
                    return (take_profit - entry_price) / entry_price
                elif exit_price <= stop_loss:
                    return (stop_loss - entry_price) / entry_price
            else:
                if exit_price <= take_profit:
                    return (entry_price - take_profit) / entry_price
                elif exit_price >= stop_loss:
                    return (entry_price - stop_loss) / entry_price
                    
            return profit
            
        except Exception as e:
            logger.error(f"盈亏计算失败: {e}")
            return 0.0 
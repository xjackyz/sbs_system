import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import torch
import logging

from src.utils.logger import setup_logger

logger = setup_logger('adaptive_learner')

@dataclass
class MarketCondition:
    """市场条件数据类"""
    volatility: float
    trend_strength: float
    liquidity: float
    timestamp: str = datetime.now().isoformat()

@dataclass
class AdaptiveConfig:
    """自适应配置"""
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.6
    liquidity_threshold: float = 0.5
    learning_rate_bounds: tuple = (1e-6, 1e-3)
    batch_size_bounds: tuple = (16, 128)
    update_interval: int = 100

class AdaptiveLearner:
    """自适应学习器"""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        初始化自适应学习器
        
        Args:
            config: 自适应配置
        """
        self.config = config or AdaptiveConfig()
        self.market_conditions = []
        self.performance_history = []
        self.current_parameters = {}
        
    def detect_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """
        检测市场条件
        
        Args:
            data: 市场数据
            
        Returns:
            市场条件
        """
        try:
            # 计算波动率
            returns = data['close'].pct_change()
            volatility = returns.std()
            
            # 计算趋势强度
            ma20 = data['close'].rolling(20).mean()
            ma50 = data['close'].rolling(50).mean()
            trend_strength = (ma20 - ma50).abs().mean() / data['close'].mean()
            
            # 计算流动性
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].tail(20).mean()
            liquidity = recent_volume / avg_volume
            
            condition = MarketCondition(
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                liquidity=float(liquidity)
            )
            
            self.market_conditions.append(condition)
            return condition
            
        except Exception as e:
            logger.error(f"市场条件检测失败: {e}")
            return MarketCondition(0.0, 0.0, 0.0)
            
    def adjust_parameters(self, condition: MarketCondition) -> Dict:
        """
        根据市场条件调整参数
        
        Args:
            condition: 市场条件
            
        Returns:
            调整后的参数
        """
        try:
            params = {}
            
            # 根据波动率调整学习率
            if condition.volatility > self.config.volatility_threshold:
                lr = np.clip(
                    condition.volatility * 0.1,
                    *self.config.learning_rate_bounds
                )
                params['learning_rate'] = float(lr)
            
            # 根据趋势强度调整批处理大小
            if condition.trend_strength > self.config.trend_threshold:
                batch_size = int(np.clip(
                    64 * condition.trend_strength,
                    *self.config.batch_size_bounds
                ))
                params['batch_size'] = batch_size
            
            self.current_parameters = params
            return params
            
        except Exception as e:
            logger.error(f"参数调整失败: {e}")
            return {}
            
    def select_strategy(self, condition: MarketCondition) -> str:
        """
        根据市场条件选择策略
        
        Args:
            condition: 市场条件
            
        Returns:
            策略名称
        """
        try:
            if condition.volatility > self.config.volatility_threshold:
                return 'conservative'
            elif condition.trend_strength > self.config.trend_threshold:
                return 'aggressive'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"策略选择失败: {e}")
            return 'neutral'
            
    def update_online(self, 
                     inputs: torch.Tensor,
                     targets: torch.Tensor,
                     model: torch.nn.Module) -> Dict:
        """
        在线学习更新
        
        Args:
            inputs: 输入数据
            targets: 目标数据
            model: 模型
            
        Returns:
            更新结果
        """
        try:
            # 计算当前性能
            with torch.no_grad():
                outputs = model(inputs)
                initial_loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # 执行一步梯度更新
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.current_parameters.get('learning_rate', 1e-4)
            )
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 记录更新结果
            result = {
                'initial_loss': float(initial_loss),
                'final_loss': float(loss),
                'improvement': float(initial_loss - loss),
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"在线学习更新失败: {e}")
            return {}
            
    def process_feedback(self, feedback: Dict) -> Dict:
        """
        处理反馈信息
        
        Args:
            feedback: 反馈信息
            
        Returns:
            处理结果
        """
        try:
            # 分析反馈
            success_rate = feedback.get('success_rate', 0.0)
            avg_return = feedback.get('average_return', 0.0)
            
            # 根据反馈调整参数
            if success_rate < 0.5:
                self.current_parameters['learning_rate'] *= 0.9
            else:
                self.current_parameters['learning_rate'] *= 1.1
                
            # 限制参数范围
            self.current_parameters['learning_rate'] = np.clip(
                self.current_parameters['learning_rate'],
                *self.config.learning_rate_bounds
            )
            
            return {
                'adjusted_parameters': self.current_parameters,
                'feedback_metrics': {
                    'success_rate': success_rate,
                    'average_return': avg_return
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"反馈处理失败: {e}")
            return {}
            
    def track_performance(self) -> Dict:
        """
        追踪性能指标
        
        Returns:
            性能统计
        """
        try:
            if not self.performance_history:
                return {}
                
            recent_history = self.performance_history[-self.config.update_interval:]
            
            stats = {
                'average_loss': np.mean([h['final_loss'] for h in recent_history]),
                'average_improvement': np.mean([h['improvement'] for h in recent_history]),
                'success_rate': np.mean([h['improvement'] > 0 for h in recent_history]),
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"性能追踪失败: {e}")
            return {}
            
    def save_state(self, path: str):
        """保存学习器状态"""
        try:
            state = {
                'market_conditions': self.market_conditions,
                'performance_history': self.performance_history,
                'current_parameters': self.current_parameters,
                'config': self.config.__dict__
            }
            
            torch.save(state, path)
            logger.info(f"学习器状态已保存到: {path}")
            
        except Exception as e:
            logger.error(f"状态保存失败: {e}")
            
    def load_state(self, path: str):
        """加载学习器状态"""
        try:
            state = torch.load(path)
            
            self.market_conditions = state['market_conditions']
            self.performance_history = state['performance_history']
            self.current_parameters = state['current_parameters']
            self.config = AdaptiveConfig(**state['config'])
            
            logger.info(f"学习器状态已从{path}加载")
            
        except Exception as e:
            logger.error(f"状态加载失败: {e}") 
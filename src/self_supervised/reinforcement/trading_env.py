import numpy as np
from typing import Dict, Tuple, Optional
import gym
from gym import spaces
import pandas as pd
from dataclasses import dataclass
import logging

from src.self_supervised.validator.sequence_evaluator import SequenceEvaluator, RewardConfig

logger = logging.getLogger(__name__)

@dataclass
class EnvConfig:
    """环境配置"""
    window_size: int = 100  # 观察窗口大小
    max_steps: int = 1000   # 最大步数
    initial_balance: float = 10000.0  # 初始资金
    transaction_fee: float = 0.001    # 交易费率

class TradingEnvironment(gym.Env):
    """交易环境"""
    
    def __init__(self, data: pd.DataFrame, config: Optional[EnvConfig] = None,
                 reward_config: Optional[RewardConfig] = None):
        """
        初始化环境
        
        Args:
            data: 市场数据
            config: 环境配置
            reward_config: 奖励配置
        """
        super().__init__()
        
        self.data = data
        self.config = config or EnvConfig()
        
        # 创建序列评估器
        self.evaluator = SequenceEvaluator(reward_config)
        
        # 定义动作空间
        # 0: 不操作, 1: 买入, 2: 卖出
        self.action_space = spaces.Discrete(3)
        
        # 定义观察空间
        # [价格数据, 技术指标, 账户状态]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, 7),  # OHLCV + 2个技术指标
            dtype=np.float32
        )
        
        # 初始化状态
        self.reset()
        
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观察
        """
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0  # 持仓数量
        self.entry_price = 0.0  # 入场价格
        
        # 交易统计
        self.trades = []
        self.total_reward = 0
        self.episode_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        return self._get_observation()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作(0: 不操作, 1: 买入, 2: 卖出)
            
        Returns:
            observation: 新的观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_step += 1
        
        # 获取当前价格
        current_price = self.data.iloc[self.current_step]['close']
        
        # 执行交易
        reward = 0
        info = {}
        
        if action == 1 and self.position == 0:  # 买入
            # 计算可买数量
            quantity = self.balance / current_price
            fee = quantity * current_price * self.config.transaction_fee
            
            if self.balance >= fee:
                self.position = quantity
                self.entry_price = current_price
                self.balance -= (quantity * current_price + fee)
                
                # 评估买入决策
                prediction = self._get_current_prediction()
                reward, metrics = self.evaluator.evaluate_breakout(
                    {'price_data': self._get_price_data()},
                    prediction
                )
                info['trade'] = {
                    'type': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'fee': fee
                }
                info['metrics'] = metrics
                
        elif action == 2 and self.position > 0:  # 卖出
            # 计算收益
            fee = self.position * current_price * self.config.transaction_fee
            profit = self.position * (current_price - self.entry_price) - fee
            
            self.balance += (self.position * current_price - fee)
            self.position = 0
            
            # 记录交易
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'profit': profit,
                'fee': fee
            })
            
            # 评估卖出决策
            prediction = self._get_current_prediction()
            reward, metrics = self.evaluator.evaluate_trend_continuation(
                {'price_data': self._get_price_data()},
                prediction
            )
            info['trade'] = {
                'type': 'sell',
                'price': current_price,
                'profit': profit,
                'fee': fee
            }
            info['metrics'] = metrics
            
            # 更新统计
            self.episode_stats['total_trades'] += 1
            if profit > 0:
                self.episode_stats['profitable_trades'] += 1
            self.episode_stats['total_profit'] += profit
            
        # 更新最大回撤
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        drawdown = (self.config.initial_balance - portfolio_value) / self.config.initial_balance
        self.episode_stats['max_drawdown'] = max(self.episode_stats['max_drawdown'], drawdown)
        
        # 累计奖励
        self.total_reward += reward
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1 or self.current_step >= self.config.max_steps
        
        # 如果结束，添加额外信息
        if done:
            info['episode'] = {
                'total_reward': self.total_reward,
                'total_trades': self.episode_stats['total_trades'],
                'profitable_trades': self.episode_stats['profitable_trades'],
                'total_profit': self.episode_stats['total_profit'],
                'max_drawdown': self.episode_stats['max_drawdown'],
                'final_balance': self.balance
            }
            
        return self._get_observation(), reward, done, info
        
    def _get_observation(self) -> np.ndarray:
        """获取观察数据"""
        # 获取价格窗口
        start = max(0, self.current_step - self.config.window_size + 1)
        end = self.current_step + 1
        
        # 提取OHLCV数据
        window_data = self.data.iloc[start:end]
        
        # 添加技术指标
        observation = np.column_stack((
            window_data[['open', 'high', 'low', 'close', 'volume']].values,
            window_data['SMA20'].values,
            window_data['SMA200'].values
        ))
        
        # 填充缺失数据
        if len(observation) < self.config.window_size:
            padding = np.zeros((self.config.window_size - len(observation), 7))
            observation = np.vstack((padding, observation))
            
        return observation.astype(np.float32)
        
    def _get_price_data(self) -> np.ndarray:
        """获取价格数据"""
        start = max(0, self.current_step - self.config.window_size + 1)
        end = self.current_step + 1
        return self.data.iloc[start:end]['close'].values
        
    def _get_current_prediction(self) -> Dict:
        """获取当前预测"""
        # 这里应该根据实际模型输出构建预测字典
        # 目前返回一个示例结构
        return {
            'breakout': {
                'index': self.current_step,
                'price': self.data.iloc[self.current_step]['close'],
                'direction': 'up' if self.position > 0 else 'down',
                'confidence': 0.8
            },
            'point1': None,  # 在实际应用中应该由模型预测
            'point2': None,
            'point3': None,
            'point4': None,
            'point5': None,
            'take_profit': self.entry_price * 1.02 if self.position > 0 else 0.0,
            'stop_loss': self.entry_price * 0.98 if self.position > 0 else 0.0
        }
        
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Total Reward: {self.total_reward:.2f}")
            if self.position > 0:
                print(f"Entry Price: {self.entry_price:.2f}")
                current_price = self.data.iloc[self.current_step]['close']
                profit = self.position * (current_price - self.entry_price)
                print(f"Current Profit: {profit:.2f}")
                
    def close(self):
        """关闭环境"""
        pass 
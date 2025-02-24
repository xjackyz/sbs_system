import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import logging
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_supervised.reinforcement.ppo_agent import PPOAgent, PPOConfig
from src.self_supervised.reinforcement.trading_env import TradingEnvironment, EnvConfig
from src.self_supervised.validator.sequence_evaluator import RewardConfig
from src.utils.logger import setup_logger

logger = setup_logger('rl_trainer')

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, data_path: str, model_save_dir: str = 'models/rl'):
        """
        初始化训练器
        
        Args:
            data_path: 数据文件路径
            model_save_dir: 模型保存目录
        """
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        
        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 加载数据
        self.data = self._load_data()
        
        # 配置
        self.ppo_config = PPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_epochs=4,
            batch_size=64
        )
        
        self.env_config = EnvConfig(
            window_size=100,
            max_steps=1000,
            initial_balance=10000.0,
            transaction_fee=0.001
        )
        
        self.reward_config = RewardConfig(
            breakout_reward=1.0,
            false_breakout_penalty=-1.0,
            point1_reward=1.0,
            point2_reward=1.0,
            point3_reward=1.0,
            point4_reward=1.0,
            point5_reward=1.0,
            invalid_sequence_penalty=-2.0,
            early_exit_penalty=-1.0,
            successful_trade_reward=2.0
        )
        
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            data = pd.read_csv(self.data_path)
            
            # 确保datetime列存在
            if 'datetime' not in data.columns:
                data['datetime'] = pd.to_datetime(data.iloc[:, 0])
            else:
                data['datetime'] = pd.to_datetime(data['datetime'])
                
            # 设置datetime为索引
            data.set_index('datetime', inplace=True)
            
            # 计算技术指标
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA200'] = data['close'].rolling(window=200).mean()
            
            # 删除NaN值
            data.dropna(inplace=True)
            
            logger.info(f"数据加载完成，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
            
    def train(self, num_episodes: int = 1000, eval_interval: int = 10,
             save_interval: int = 100) -> Dict[str, List[float]]:
        """
        训练代理
        
        Args:
            num_episodes: 训练轮数
            eval_interval: 评估间隔
            save_interval: 保存间隔
            
        Returns:
            训练历史
        """
        try:
            # 创建环境
            env = TradingEnvironment(
                data=self.data,
                config=self.env_config,
                reward_config=self.reward_config
            )
            
            # 创建代理
            state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_dim = env.action_space.n
            agent = PPOAgent(state_dim, action_dim, self.ppo_config)
            
            # 训练历史
            history = {
                'episode_rewards': [],
                'total_profits': [],
                'win_rates': [],
                'max_drawdowns': []
            }
            
            logger.info("开始训练...")
            
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # 将状态展平
                    flat_state = state.reshape(-1)
                    
                    # 选择动作
                    action, value, log_prob = agent.select_action(flat_state)
                    
                    # 执行动作
                    next_state, reward, done, info = env.step(action)
                    
                    # 存储转换
                    agent.store_transition(
                        flat_state,
                        action,
                        reward,
                        value,
                        log_prob,
                        0.0 if done else 1.0
                    )
                    
                    state = next_state
                    episode_reward += reward
                    
                # 更新策略
                if len(agent.states) > 0:
                    next_value = 0 if done else agent.network(
                        torch.FloatTensor(state.reshape(-1)).unsqueeze(0).to(agent.device)
                    )[1].item()
                    metrics = agent.update(next_value)
                    
                # 记录历史
                history['episode_rewards'].append(episode_reward)
                if 'episode' in info:
                    history['total_profits'].append(info['episode']['total_profit'])
                    win_rate = (info['episode']['profitable_trades'] / 
                              max(1, info['episode']['total_trades']))
                    history['win_rates'].append(win_rate)
                    history['max_drawdowns'].append(info['episode']['max_drawdown'])
                    
                # 打印进度
                if (episode + 1) % eval_interval == 0:
                    mean_reward = np.mean(history['episode_rewards'][-eval_interval:])
                    mean_profit = np.mean(history['total_profits'][-eval_interval:])
                    mean_win_rate = np.mean(history['win_rates'][-eval_interval:])
                    mean_drawdown = np.mean(history['max_drawdowns'][-eval_interval:])
                    
                    logger.info(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"平均奖励: {mean_reward:.2f} | "
                        f"平均利润: {mean_profit:.2f} | "
                        f"胜率: {mean_win_rate:.2%} | "
                        f"最大回撤: {mean_drawdown:.2%}"
                    )
                    
                # 保存模型
                if (episode + 1) % save_interval == 0:
                    save_path = os.path.join(
                        self.model_save_dir,
                        f"ppo_agent_episode_{episode + 1}.pt"
                    )
                    agent.save(save_path)
                    logger.info(f"模型已保存到: {save_path}")
                    
            logger.info("训练完成")
            return history
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
            
    def evaluate(self, model_path: str, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            model_path: 模型路径
            num_episodes: 评估轮数
            
        Returns:
            评估指标
        """
        try:
            # 创建环境
            env = TradingEnvironment(
                data=self.data,
                config=self.env_config,
                reward_config=self.reward_config
            )
            
            # 加载代理
            state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_dim = env.action_space.n
            agent = PPOAgent(state_dim, action_dim, self.ppo_config)
            agent.load(model_path)
            
            # 评估指标
            total_rewards = []
            total_profits = []
            win_rates = []
            max_drawdowns = []
            
            logger.info("开始评估...")
            
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # 选择动作
                    action, _, _ = agent.select_action(state.reshape(-1))
                    
                    # 执行动作
                    state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                # 记录指标
                total_rewards.append(episode_reward)
                if 'episode' in info:
                    total_profits.append(info['episode']['total_profit'])
                    win_rate = (info['episode']['profitable_trades'] / 
                              max(1, info['episode']['total_trades']))
                    win_rates.append(win_rate)
                    max_drawdowns.append(info['episode']['max_drawdown'])
                    
            # 计算平均指标
            metrics = {
                'mean_reward': np.mean(total_rewards),
                'mean_profit': np.mean(total_profits),
                'mean_win_rate': np.mean(win_rates),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'std_reward': np.std(total_rewards),
                'std_profit': np.std(total_profits)
            }
            
            logger.info(
                f"评估完成 | "
                f"平均奖励: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} | "
                f"平均利润: {metrics['mean_profit']:.2f} ± {metrics['std_profit']:.2f} | "
                f"平均胜率: {metrics['mean_win_rate']:.2%} | "
                f"平均最大回撤: {metrics['mean_max_drawdown']:.2%}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            raise

def main():
    """主函数"""
    try:
        # 设置参数
        data_path = '/home/easyai/桌面/nq/NQ_full_1min_continuous.csv'
        model_save_dir = 'models/rl'
        num_episodes = 1000
        eval_interval = 10
        save_interval = 100
        
        # 创建训练器
        trainer = RLTrainer(data_path, model_save_dir)
        
        # 训练模型
        history = trainer.train(num_episodes, eval_interval, save_interval)
        
        # 评估最后保存的模型
        last_model_path = os.path.join(model_save_dir, f"ppo_agent_episode_{num_episodes}.pt")
        if os.path.exists(last_model_path):
            trainer.evaluate(last_model_path)
            
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
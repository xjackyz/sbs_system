import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO配置"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    
class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        super().__init__()
        
        # 共享特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            action_probs: 动作概率
            value: 状态价值
        """
        features = self.feature_net(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOAgent:
    """PPO强化学习代理"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[PPOConfig] = None):
        """
        初始化代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: PPO配置
        """
        self.config = config or PPOConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建Actor-Critic网络
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # 经验回放缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        选择动作
        
        Args:
            state: 状态数组
            
        Returns:
            action: 选择的动作
            value: 状态价值
            log_prob: 动作对数概率
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state)
            
            # 采样动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), value.item(), log_prob.item()
            
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, mask: float):
        """
        存储转换
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            value: 价值
            log_prob: 对数概率
            mask: 掩码(是否终止)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        
    def update(self, next_value: float) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            next_value: 下一个状态的价值
            
        Returns:
            训练指标
        """
        # 计算优势和回报
        returns = self._compute_returns(next_value)
        advantages = returns - torch.tensor(self.values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # PPO更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.num_epochs):
            for batch_indices in self._get_minibatch_indices():
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 前向传播
                action_probs, values = self.network(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon) * batch_advantages
                
                # 计算损失
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
        # 清空缓冲区
        self._clear_memory()
        
        # 计算平均损失
        num_updates = self.config.num_epochs * (len(states) // self.config.batch_size)
        metrics = {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return metrics
        
    def _compute_returns(self, next_value: float) -> torch.Tensor:
        """计算回报"""
        returns = []
        R = next_value
        for r, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = r + self.config.gamma * R * mask
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)
        
    def _get_minibatch_indices(self) -> List[torch.Tensor]:
        """获取小批次索引"""
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        batch_size = self.config.batch_size
        minibatches = [
            torch.tensor(indices[i:i + batch_size])
            for i in range(0, len(indices), batch_size)
        ]
        return minibatches
        
    def _clear_memory(self):
        """清空经验回放缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config'] 
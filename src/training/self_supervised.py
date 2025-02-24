"""
自监督学习模块 - 非交易时段的模型训练
"""
import os
import torch
import logging
from datetime import datetime, time
import asyncio
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
import json

from ..utils.logger import setup_logger
from ..model.sbs_analyzer import SBSAnalyzer

logger = setup_logger('self_supervised')

class SignalTracker:
    """信号跟踪器"""
    
    def __init__(self, window_size: int = 5):
        """初始化信号跟踪器
        
        Args:
            window_size: 跟踪窗口大小（天）
        """
        self.window_size = window_size
        self.signals = []
        self.rewards = {}
        
    def add_signal(self, signal: Dict):
        """添加新信号
        
        Args:
            signal: 信号数据
        """
        signal['timestamp'] = datetime.now()
        signal['status'] = 'pending'
        self.signals.append(signal)
        
    def update_signal(self, signal_id: str, performance: Dict):
        """更新信号表现
        
        Args:
            signal_id: 信号ID
            performance: 表现数据
        """
        for signal in self.signals:
            if signal['id'] == signal_id:
                signal['performance'] = performance
                signal['status'] = 'completed'
                self._calculate_reward(signal)
                break
                
    def _calculate_reward(self, signal: Dict):
        """计算信号奖励
        
        Args:
            signal: 信号数据
        """
        performance = signal.get('performance', {})
        
        # 基础奖励
        if performance.get('reached_target', False):
            reward = 1.0
        elif performance.get('hit_stop_loss', False):
            reward = -1.0
        else:
            # 根据盈亏比例计算奖励
            pnl_ratio = performance.get('pnl_ratio', 0)
            reward = max(-1.0, min(1.0, pnl_ratio))
            
        # 考虑置信度
        confidence = signal.get('confidence', 0.5)
        reward *= confidence
        
        # 应用时间衰减
        days_passed = (datetime.now() - signal['timestamp']).days
        decay = 0.95 ** days_passed
        reward *= decay
        
        self.rewards[signal['id']] = reward

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model_path: str):
        """初始化模型优化器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.optimizer = None
        
    async def initialize(self):
        """初始化模型和优化器"""
        try:
            self.model = SBSAnalyzer(
                base_model=self.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 设置优化器
            trainable_params = filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            )
            self.optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
            
        except Exception as e:
            logger.error(f"模型优化器初始化失败: {e}")
            raise
            
    def update_model(self, signals: List[Dict]):
        """更新模型
        
        Args:
            signals: 信号列表
        """
        if not signals:
            return
            
        try:
            # 收集训练数据
            features = []
            rewards = []
            
            for signal in signals:
                if signal['status'] == 'completed':
                    features.append(signal['features'])
                    rewards.append(signal.get('reward', 0))
                    
            if not features:
                return
                
            # 转换为张量
            features = torch.stack(features)
            rewards = torch.tensor(rewards)
            
            # 更新模型
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self._calculate_loss(outputs, rewards)
            loss.backward()
            self.optimizer.step()
            
            logger.info(f"模型更新完成，损失: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            
    def _calculate_loss(self, outputs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            rewards: 奖励值
            
        Returns:
            torch.Tensor: 损失值
        """
        # 使用MSE损失
        return torch.nn.functional.mse_loss(outputs, rewards)
        
    def save_checkpoint(self):
        """保存检查点"""
        try:
            checkpoint_path = Path(self.model_path) / 'checkpoints' / f'model_{datetime.now():%Y%m%d_%H%M}.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            logger.info(f"模型检查点已保存: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

class ValidationManager:
    """验证集管理器"""
    
    def __init__(self, validation_dir: str):
        """初始化验证集管理器
        
        Args:
            validation_dir: 验证数据目录
        """
        self.validation_dir = Path(validation_dir)
        self.validation_data = []
        
    def load_validation_data(self):
        """加载验证数据"""
        try:
            # 加载验证集图片
            for img_path in self.validation_dir.glob("*.png"):
                self.validation_data.append({
                    'image_path': str(img_path),
                    'metadata': self._load_metadata(img_path)
                })
            logger.info(f"已加载 {len(self.validation_data)} 个验证样本")
        except Exception as e:
            logger.error(f"加载验证数据失败: {e}")
            
    def _load_metadata(self, image_path: Path) -> Dict:
        """加载图片元数据
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 元数据
        """
        try:
            metadata_path = image_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return {}
            
    def validate_model(self, model: SBSAnalyzer) -> Dict:
        """验证模型性能
        
        Args:
            model: SBS分析器实例
            
        Returns:
            Dict: 验证结果
        """
        try:
            results = {
                'total_samples': len(self.validation_data),
                'successful_predictions': 0,
                'average_confidence': 0.0,
                'false_positives': 0,
                'false_negatives': 0
            }
            
            for sample in self.validation_data:
                prediction = model.process_image(sample['image_path'])
                actual = sample['metadata'].get('actual_outcome', {})
                
                # 评估预测结果
                if self._evaluate_prediction(prediction, actual):
                    results['successful_predictions'] += 1
                    
                results['average_confidence'] += prediction.get('confidence', 0)
                
                if prediction.get('signal_type') != actual.get('signal_type'):
                    if prediction.get('signal_type') == 'no_signal':
                        results['false_negatives'] += 1
                    else:
                        results['false_positives'] += 1
                        
            # 计算平均值
            results['average_confidence'] /= len(self.validation_data)
            results['accuracy'] = results['successful_predictions'] / len(self.validation_data)
            
            logger.info(f"验证结果: {results}")
            return results
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return {}
            
    def _evaluate_prediction(self, prediction: Dict, actual: Dict) -> bool:
        """评估预测结果
        
        Args:
            prediction: 预测结果
            actual: 实际结果
            
        Returns:
            bool: 是否预测成功
        """
        # 信号类型匹配
        if prediction.get('signal_type') != actual.get('signal_type'):
            return False
            
        # 价格目标匹配（允许10%的误差）
        pred_target = prediction.get('target_price', 0)
        actual_target = actual.get('target_price', 0)
        if abs(pred_target - actual_target) / actual_target > 0.1:
            return False
            
        return True

class SelfSupervisedTrainer:
    """自监督学习训练器"""
    
    def __init__(self, config: Dict):
        """初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.signal_tracker = SignalTracker(
            window_size=config.get('signal_tracking_window', 5)
        )
        self.model_optimizer = ModelOptimizer(
            model_path=config['model_path']
        )
        self.validation_manager = ValidationManager(
            validation_dir=config['validation_dir']
        )
        self.is_training = False
        
    async def initialize(self):
        """初始化训练器"""
        try:
            await self.model_optimizer.initialize()
            self.validation_manager.load_validation_data()
            logger.info("自监督学习训练器初始化完成")
        except Exception as e:
            logger.error(f"训练器初始化失败: {e}")
            raise
            
    def is_trading_hours(self) -> bool:
        """检查是否在交易时段
        
        Returns:
            bool: 是否在交易时段
        """
        now = datetime.now().time()
        # 定义交易时段（以亚洲市场为例）
        morning_session = (time(9, 0), time(11, 30))
        afternoon_session = (time(13, 0), time(15, 0))
        
        return (morning_session[0] <= now <= morning_session[1] or
                afternoon_session[0] <= now <= afternoon_session[1])
                
    async def run(self):
        """运行训练循环"""
        try:
            await self.initialize()
            self.is_training = True
            
            while self.is_training:
                # 检查是否在交易时段
                if not self.is_trading_hours():
                    # 更新信号状态
                    self._update_signals()
                    
                    # 训练模型
                    self.model_optimizer.update_model(self.signal_tracker.signals)
                    
                    # 验证模型
                    validation_results = self.validation_manager.validate_model(
                        self.model_optimizer.model
                    )
                    
                    # 如果验证结果良好，保存检查点
                    if validation_results.get('accuracy', 0) > 0.7:
                        self.model_optimizer.save_checkpoint()
                        
                    # 等待下一次训练
                    await asyncio.sleep(3600)  # 1小时
                else:
                    # 交易时段，暂停训练
                    logger.info("当前是交易时段，暂停训练")
                    await asyncio.sleep(1800)  # 30分钟
                    
        except Exception as e:
            logger.error(f"训练循环异常: {e}")
            self.is_training = False
            
    def _update_signals(self):
        """更新信号状态"""
        try:
            for signal in self.signal_tracker.signals:
                if signal['status'] == 'pending':
                    # 检查是否超过跟踪窗口
                    days_passed = (datetime.now() - signal['timestamp']).days
                    if days_passed >= self.signal_tracker.window_size:
                        # 获取市场数据并评估信号表现
                        performance = self._evaluate_signal_performance(signal)
                        self.signal_tracker.update_signal(signal['id'], performance)
                        
        except Exception as e:
            logger.error(f"更新信号状态失败: {e}")
            
    def _evaluate_signal_performance(self, signal: Dict) -> Dict:
        """评估信号表现
        
        Args:
            signal: 信号数据
            
        Returns:
            Dict: 表现评估结果
        """
        try:
            # TODO: 实现实际的信号表现评估逻辑
            # 这里需要集成市场数据源，获取价格历史
            # 并根据实际价格走势评估信号的表现
            return {
                'reached_target': False,
                'hit_stop_loss': False,
                'pnl_ratio': 0.0
            }
        except Exception as e:
            logger.error(f"评估信号表现失败: {e}")
            return {}
            
    def stop(self):
        """停止训练"""
        self.is_training = False
        logger.info("训练已停止")

async def main():
    """主函数"""
    try:
        config = {
            'model_path': os.getenv('MODEL_PATH', 'models/llava-sbs'),
            'validation_dir': os.getenv('VALIDATION_DATA_DIR', 'data/validation'),
            'signal_tracking_window': int(os.getenv('SIGNAL_TRACKING_WINDOW', '5')),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.8')),
            'reward_decay_rate': float(os.getenv('REWARD_DECAY_RATE', '0.95')),
            'training_frequency': int(os.getenv('TRAINING_FREQUENCY', '24'))
        }
        
        trainer = SelfSupervisedTrainer(config)
        await trainer.run()
        
    except Exception as e:
        logger.error(f"自监督学习服务异常: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
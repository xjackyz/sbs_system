import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import os
import logging
from datetime import datetime
import json
import numpy as np

from src.self_supervised.data.data_loader import DataLoaderFactory
from src.self_supervised.validator.sequence_validator import SequenceValidator
from src.self_supervised.visualization.sequence_visualizer import SequenceVisualizer
from src.self_supervised.reporting.report_generator import ReportGenerator
from src.utils.memory_monitor import MemoryMonitor
from src.utils.logger import setup_logger
from config.config import (
    SELF_SUPERVISED_CONFIG,
    TRAINING_OPTIMIZATION,
    OPTIMIZER_CONFIG,
    TRAINING_WORKFLOW
)
from src.self_supervised.utils.output_formatter import OutputFormatter, OutputRequirements

logger = setup_logger('self_supervised_trainer')

class SelfSupervisedTrainer:
    """自监督学习训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        data_dir: str,
        save_dir: str,
        device: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            data_dir: 数据目录
            save_dir: 保存目录
            device: 运行设备
        """
        self.model = model
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
        
    def _init_components(self):
        """初始化组件"""
        # 创建数据加载器
        dataloaders = DataLoaderFactory.create_dataloaders(self.data_dir)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 创建验证器
        self.validator = SequenceValidator()
        
        # 创建可视化器
        self.visualizer = SequenceVisualizer(os.path.join(self.save_dir, 'figures'))
        
        # 创建报告生成器
        self.report_generator = ReportGenerator(os.path.join(self.save_dir, 'reports'))
        
        # 创建输出格式化器
        self.output_formatter = OutputFormatter(
            requirements=OutputRequirements(
                sequence_validity_threshold=SELF_SUPERVISED_CONFIG['evaluation']['validity_threshold'],
                confidence_threshold=SELF_SUPERVISED_CONFIG['evaluation']['confidence_threshold']
            )
        )
        
        # 训练历史
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # 早停参数
        self.best_val_loss = float('inf')
        self.patience = SELF_SUPERVISED_CONFIG['training']['early_stopping_patience']
        self.patience_counter = 0
        
        # 创建内存监控器
        self.memory_monitor = MemoryMonitor()
        
        # 创建梯度缩放器(用于混合精度训练)
        self.scaler = GradScaler() if TRAINING_OPTIMIZATION['mixed_precision'] else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.training_history = []
        
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_class = getattr(optim, OPTIMIZER_CONFIG['type'])
        return optimizer_class(
            self.model.parameters(),
            lr=OPTIMIZER_CONFIG['lr'],
            weight_decay=OPTIMIZER_CONFIG['weight_decay']
        )
        
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        scheduler_config = OPTIMIZER_CONFIG['scheduler']
        scheduler_class = getattr(optim.lr_scheduler, scheduler_config['type'])
        
        return scheduler_class(
            self.optimizer,
            max_lr=scheduler_config['max_lr'],
            pct_start=scheduler_config['pct_start'],
            div_factor=scheduler_config['div_factor'],
            final_div_factor=scheduler_config['final_div_factor']
        )
        
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        return nn.CrossEntropyLoss()
        
    def train(self, num_epochs: Optional[int] = None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
        """
        num_epochs = num_epochs or SELF_SUPERVISED_CONFIG['training']['max_epochs']
        logger.info(f"开始训练，总轮数: {num_epochs}")
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"轮次 {epoch+1}/{num_epochs}")
                
                # 训练一个轮次
                train_metrics = self._train_epoch()
                
                # 验证
                val_metrics = self._validate()
                
                # 更新学习率
                self.scheduler.step(val_metrics['loss'])
                
                # 更新历史记录
                self._update_history(train_metrics, val_metrics)
                
                # 保存检查点
                self._save_checkpoint(val_metrics)
                
                # 记录进度
                self._log_progress(train_metrics, val_metrics)
                
                # 检查是否早停
                if self._should_stop_early():
                    logger.info("触发早停")
                    break
                    
            # 保存训练历史
            self._save_history()
            
            # 生成训练报告
            self.report_generator.generate_training_report(
                training_history=self.history,
                evaluation_results=self.evaluate(),
                model_config=self.model.config.__dict__,
                save_name='final_training_report.html'
            )
            
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            raise
            
    def _train_epoch(self) -> Dict:
        """训练一个轮次"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            try:
                # 移动数据到设备
                inputs = self._prepare_inputs(inputs)
                targets = self._prepare_targets(targets)
                
                # 前向传播
                with autocast(enabled=TRAINING_OPTIMIZATION['mixed_precision']):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs['sequence_logits'], targets)
                    
                # 反向传播
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                    
                # 统计
                total_loss += loss.item()
                total_samples += targets.size(0)
                
                # 计算准确率
                _, predicted = outputs['sequence_logits'].max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
                # 记录当前学习率
                self.history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # 更新进度条
                self._log_progress(
                    {
                        'loss': loss.item(),
                        'accuracy': correct_predictions / total_samples
                    },
                    {
                        'loss': loss.item(),
                        'accuracy': correct_predictions / total_samples
                    }
                )
                
                # 更新全局步数
                self.global_step += 1
                
                # 检查内存使用
                if batch_idx % 10 == 0:
                    self.memory_monitor.check_memory()
                    
            except Exception as e:
                logger.error(f"训练批次[{batch_idx}]出错: {e}")
                continue
                
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct_predictions / total_samples
        }
        
    def _validate(self) -> Dict:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        validation_results = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                try:
                    # 移动数据到设备
                    inputs = self._prepare_inputs(inputs)
                    targets = self._prepare_targets(targets)
                    
                    # 前向传播
                    with autocast(enabled=TRAINING_OPTIMIZATION['mixed_precision']):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs['sequence_logits'], targets)
                        
                    # 统计
                    total_loss += loss.item()
                    total_samples += targets.size(0)
                    
                    # 计算准确率
                    _, predicted = outputs['sequence_logits'].max(1)
                    correct_predictions += predicted.eq(targets).sum().item()
                    
                    # 序列验证
                    for i in range(len(inputs['sequence_data'])):
                        validation_result = self.validator.validate_sequence(
                            inputs['sequence_data'][i],
                            {
                                'breakout_index': outputs['point_coords'][i][0].tolist(),
                                'direction': 'up' if predicted[i] == 1 else 'down',
                                'point1_index': outputs['point_coords'][i][1].tolist(),
                                'point2_index': outputs['point_coords'][i][2].tolist(),
                                'point3_index': outputs['point_coords'][i][3].tolist(),
                                'point4_index': outputs['point_coords'][i][4].tolist(),
                                'point5_index': outputs['point_coords'][i][5].tolist()
                            }
                        )
                        validation_results.append(validation_result)
                    
                except Exception as e:
                    logger.error(f"验证过程出错: {e}")
                    continue
                    
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct_predictions / total_samples,
            'validation_results': validation_results
        }
        
    def _prepare_inputs(self, inputs: Dict) -> Dict:
        """准备输入数据"""
        return {
            'chart_image': inputs['chart_image'].to(self.device),
            'sequence_data': inputs['sequence_data'].to(self.device)
        }
        
    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """准备目标数据"""
        return targets.to(self.device)
        
    def _save_checkpoint(self, val_metrics: Dict):
        """保存检查点"""
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.patience_counter = 0
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'config': self.model.config.__dict__
            }
            
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存最佳模型检查点: {checkpoint_path}")
            
        else:
            self.patience_counter += 1
            
    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        """更新历史记录"""
        self.history['loss'].append(train_metrics['loss'])
        self.history['accuracy'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        
    def _save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"保存训练历史: {history_path}")
        
    def _log_progress(self, train_metrics: Dict, val_metrics: Dict):
        """记录训练进度"""
        logger.info(
            f"训练损失: {train_metrics['loss']:.4f}, "
            f"训练准确率: {train_metrics['accuracy']:.4f}, "
            f"验证损失: {val_metrics['loss']:.4f}, "
            f"验证准确率: {val_metrics['accuracy']:.4f}"
        )
        
    def _should_stop_early(self) -> bool:
        """检查是否应该早停"""
        return self.patience_counter >= self.patience
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint['history']
            
            logger.info(f"成功加载检查点: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
            
    def evaluate(self) -> Dict:
        """
        评估模型
        
        Returns:
            Dict: 评估结果
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        validation_results = []
        raw_outputs = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                try:
                    # 移动数据到设备
                    inputs = self._prepare_inputs(inputs)
                    targets = self._prepare_targets(targets)
                    
                    # 前向传播
                    with autocast(enabled=TRAINING_OPTIMIZATION['mixed_precision']):
                        outputs = self.model(inputs)
                        
                    # 收集预测结果
                    _, predicted = outputs['sequence_logits'].max(1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # 序列验证
                    for i in range(len(inputs['sequence_data'])):
                        # 收集原始输出
                        raw_output = {
                            'validity_score': outputs['validity_score'][i].item(),
                            'points': [
                                {
                                    'index': idx,
                                    'price': price,
                                    'confidence': conf,
                                    'timestamp': datetime.now().isoformat()
                                }
                                for idx, (price, conf) in enumerate(zip(
                                    outputs['point_coords'][i].tolist(),
                                    outputs['point_confidences'][i].tolist()
                                ))
                            ],
                            'signals': {
                                'action': 'buy' if predicted[i] == 1 else 'sell',
                                'confidence': outputs['prediction_confidence'][i].item(),
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                        raw_outputs.append(raw_output)
                        
                        # 验证序列
                        validation_result = self.validator.validate_sequence(
                            inputs['sequence_data'][i],
                            {
                                'breakout_index': outputs['point_coords'][i][0].tolist(),
                                'direction': 'up' if predicted[i] == 1 else 'down',
                                'point1_index': outputs['point_coords'][i][1].tolist(),
                                'point2_index': outputs['point_coords'][i][2].tolist(),
                                'point3_index': outputs['point_coords'][i][3].tolist(),
                                'point4_index': outputs['point_coords'][i][4].tolist(),
                                'point5_index': outputs['point_coords'][i][5].tolist()
                            }
                        )
                        validation_results.append(validation_result)
                    
                except Exception as e:
                    logger.error(f"评估过程出错: {e}")
                    continue
                    
        # 计算评估指标
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score
        
        cm = confusion_matrix(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        accuracy = cm.diagonal().sum() / cm.sum()
        
        # 格式化原始输出
        formatted_outputs = [
            self.output_formatter.format_model_outputs(output)
            for output in raw_outputs
        ]
        
        # 准备评估结果
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'false_positives': cm[0][1],
            'false_negatives': cm[1][0],
            'validation_results': validation_results,
            'formatted_outputs': formatted_outputs
        }
        
        # 生成评估报告
        self.report_generator.generate_evaluation_report(
            validation_results=validation_results,
            sequence_data=[inputs['sequence_data'] for inputs, _ in self.test_loader],
            save_name='final_evaluation_report.html'
        )
        
        return evaluation_results 
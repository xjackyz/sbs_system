import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from .quantization import ModelQuantizer
from .pruning import ModelPruner
from .distillation import KnowledgeDistillation

class OptimizationManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantizer = ModelQuantizer(model)
        self.pruner = ModelPruner(model)
        self.distillation = None  # 知识蒸馏需要学生模型，稍后初始化
        
    def apply_quantization(self,
                          method: str = 'dynamic',
                          calibration_data: Optional[torch.Tensor] = None,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """应用量化优化
        
        Args:
            method: 量化方法 ('dynamic' 或 'static')
            calibration_data: 静态量化校准数据
            config: 量化配置
        """
        if method == 'dynamic':
            quantized_model = self.quantizer.dynamic_quantize()
        elif method == 'static':
            if calibration_data is None:
                raise ValueError("静态量化需要校准数据")
            quantized_model = self.quantizer.static_quantize(
                calibration_data=calibration_data,
                config=config
            )
        else:
            raise ValueError(f"不支持的量化方法: {method}")
            
        return self.quantizer.evaluate_performance(
            test_data=calibration_data,
            original_model=self.model
        )
        
    def apply_pruning(self,
                     layer_names: List[str],
                     method: str = 'unstructured',
                     amount: float = 0.3,
                     dim: int = 0) -> Dict[str, float]:
        """应用剪枝优化
        
        Args:
            layer_names: 需要剪枝的层名称列表
            method: 剪枝方法 ('unstructured' 或 'structured')
            amount: 剪枝比例
            dim: 结构化剪枝的维度
        """
        self.pruner.save_original_state()
        
        if method == 'unstructured':
            results = self.pruner.unstructured_pruning(
                layer_names=layer_names,
                amount=amount
            )
        elif method == 'structured':
            results = self.pruner.structured_pruning(
                layer_names=layer_names,
                amount=amount,
                dim=dim
            )
        else:
            raise ValueError(f"不支持的剪枝方法: {method}")
            
        return results
    
    def setup_distillation(self,
                          student_model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          **kwargs) -> None:
        """设置知识蒸馏
        
        Args:
            student_model: 学生模型
            optimizer: 优化器
            **kwargs: 其他知识蒸馏参数
        """
        self.distillation = KnowledgeDistillation(
            teacher_model=self.model,
            student_model=student_model,
            optimizer=optimizer,
            **kwargs
        )
        
    def train_distillation(self,
                          train_loader: torch.utils.data.DataLoader,
                          val_loader: Optional[torch.utils.data.DataLoader] = None,
                          num_epochs: int = 10,
                          save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """训练知识蒸馏模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
        """
        if self.distillation is None:
            raise ValueError("请先调用setup_distillation设置知识蒸馏")
            
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            for batch_data, labels in train_loader:
                results = self.distillation.train_step(batch_data, labels)
                epoch_loss += results['loss']
                epoch_accuracy += results['accuracy']
                num_batches += 1
                
            history['train_loss'].append(epoch_loss / num_batches)
            history['train_accuracy'].append(epoch_accuracy / num_batches)
            
            if val_loader is not None:
                val_results = self.distillation.evaluate(val_loader)
                history['val_loss'].append(val_results['val_loss'])
                history['val_accuracy'].append(val_results['val_accuracy'])
                
        if save_path:
            self.distillation.save_student_model(save_path)
            
        return history
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        summary = {
            'pruning_stats': self.pruner.get_pruning_statistics()
        }
        
        if self.distillation is not None:
            teacher_size, student_size = self.distillation.get_model_sizes()
            summary['distillation_stats'] = {
                'teacher_model_size_mb': teacher_size,
                'student_model_size_mb': student_size,
                'compression_ratio': teacher_size / student_size
            }
            
        return summary 
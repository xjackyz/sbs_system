import torch
import torch.nn.utils.prune as prune
from typing import Dict, List, Union, Optional
import numpy as np

class ModelPruner:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.original_state_dict = None
        self.pruning_history = []
        
    def save_original_state(self):
        """保存原始模型状态"""
        self.original_state_dict = self.model.state_dict()
        
    def unstructured_pruning(self, 
                           layer_names: List[str],
                           amount: float = 0.3) -> Dict[str, float]:
        """非结构化剪枝
        
        Args:
            layer_names: 需要剪枝的层名称列表
            amount: 剪枝比例（0-1之间）
        """
        results = {}
        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                prune.l1_unstructured(layer, name='weight', amount=amount)
                
                # 计算稀疏度
                sparsity = 100. * float(torch.sum(layer.weight == 0)) / float(layer.weight.nelement())
                results[name] = sparsity
                
                self.pruning_history.append({
                    'layer': name,
                    'type': 'unstructured',
                    'amount': amount,
                    'sparsity': sparsity
                })
                
        return results
    
    def structured_pruning(self,
                         layer_names: List[str],
                         amount: float = 0.3,
                         dim: int = 0) -> Dict[str, float]:
        """结构化剪枝
        
        Args:
            layer_names: 需要剪枝的层名称列表
            amount: 剪枝比例（0-1之间）
            dim: 剪枝维度（0表示按输出通道剪枝，1表示按输入通道剪枝）
        """
        results = {}
        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                prune.ln_structured(layer, name='weight', amount=amount, 
                                 n=2, dim=dim)
                
                # 计算稀疏度
                sparsity = 100. * float(torch.sum(layer.weight == 0)) / float(layer.weight.nelement())
                results[name] = sparsity
                
                self.pruning_history.append({
                    'layer': name,
                    'type': 'structured',
                    'amount': amount,
                    'dim': dim,
                    'sparsity': sparsity
                })
                
        return results
    
    def remove_pruning(self, layer_names: Optional[List[str]] = None):
        """移除剪枝，恢复原始权重"""
        if layer_names is None:
            # 移除所有层的剪枝
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
        else:
            # 移除指定层的剪枝
            for name in layer_names:
                if hasattr(self.model, name):
                    layer = getattr(self.model, name)
                    prune.remove(layer, 'weight')
                    
    def get_pruning_statistics(self) -> Dict[str, Union[float, List[Dict]]]:
        """获取剪枝统计信息"""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.nelement()
            zero_params += torch.sum(param == 0).item()
            
        return {
            'global_sparsity': 100. * zero_params / total_params,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'pruning_history': self.pruning_history
        } 
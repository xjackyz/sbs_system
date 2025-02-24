import torch
import torch.quantization
from typing import Optional, Dict, Any

class ModelQuantizer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.quantized_model = None
        
    def dynamic_quantize(self) -> torch.nn.Module:
        """动态量化模型"""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
        return self.quantized_model
    
    def static_quantize(self, 
                       calibration_data: torch.Tensor,
                       config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """静态量化模型"""
        # 准备量化配置
        if config is None:
            config = torch.quantization.get_default_qconfig('fbgemm')
            
        # 准备模型进行量化
        self.model.qconfig = config
        torch.quantization.prepare(self.model, inplace=True)
        
        # 使用校准数据进行校准
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
                
        # 完成量化
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model
    
    def evaluate_performance(self, 
                           test_data: torch.Tensor,
                           original_model: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """评估量化前后的模型性能"""
        results = {}
        
        # 测试原始模型
        if original_model is not None:
            original_size = self._get_model_size(original_model)
            results['original_size_mb'] = original_size
            
        # 测试量化后的模型
        if self.quantized_model is not None:
            quantized_size = self._get_model_size(self.quantized_model)
            results['quantized_size_mb'] = quantized_size
            results['compression_ratio'] = original_size / quantized_size if original_model is not None else None
            
        return results
    
    @staticmethod
    def _get_model_size(model: torch.nn.Module) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb 
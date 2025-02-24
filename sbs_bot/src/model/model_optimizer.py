import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from pathlib import Path
import os

from src.utils.logger import setup_logger

logger = setup_logger('model_optimizer')

class ModelOptimizer:
    def __init__(self, model_path: str, device: str = None):
        """
        初始化模型优化器
        
        Args:
            model_path: 模型路径
            device: 运行设备（'cuda'或'cpu'）
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_optimized = False
        
        logger.info(f"Model optimizer initialized with device: {self.device}")

    def optimize_model(self, batch_size: int = 1):
        """
        优化模型性能
        
        Args:
            batch_size: 批处理大小
        """
        try:
            # 暂时禁用ONNX优化
            self.is_optimized = True
            logger.info("Model optimization skipped (ONNX disabled)")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise

    @torch.no_grad()
    def inference(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        执行模型推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            Dict: 推理结果
        """
        try:
            if not self.is_optimized:
                self.optimize_model()
            
            # 直接使用PyTorch进行推理
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data)
            input_data = input_data.to(self.device)
            
            # 加载模型
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            
            # 执行推理
            output = model(input_data)
            
            result = {
                'output': output.cpu(),
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'output': None,
                'success': False,
                'error': str(e)
            }

    def cleanup(self):
        """清理资源"""
        try:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Model optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 
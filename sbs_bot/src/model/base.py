"""
模型系统基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str  # 模型名称
    version: str  # 模型版本
    description: str  # 模型描述
    input_type: str  # 输入类型(image/text/mixed)
    output_type: str  # 输出类型
    architecture: str  # 模型架构
    parameters: int  # 参数数量
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    
class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self):
        """初始化"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[ModelMetadata] = None
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化模型"""
        pass
        
    @abstractmethod
    async def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """模型推理"""
        pass
        
    @abstractmethod
    def validate_input(self, inputs: Dict[str, Any]) -> bool:
        """验证输入数据"""
        pass
        
    @abstractmethod
    def validate_output(self, outputs: Dict[str, Any]) -> bool:
        """验证输出数据"""
        pass
        
    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """获取模型元数据"""
        pass
        
    @abstractmethod
    def get_model_size(self) -> int:
        """获取模型大小(字节)"""
        pass
        
    def to(self, device: str):
        """移动模型到指定设备"""
        if self.model is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
            
    def eval(self):
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()
            
    def train(self):
        """设置为训练模式"""
        if self.model is not None:
            self.model.train()
            
    async def warmup(self):
        """模型预热"""
        pass
        
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        self.cleanup() 
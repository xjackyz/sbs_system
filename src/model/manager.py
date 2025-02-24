"""
模型管理器
"""
import os
import asyncio
from typing import Dict, Optional, Any, Type
from datetime import datetime
import torch
from .base import BaseModel, ModelMetadata
from ..config.model_config import ModelConfig

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """初始化"""
        self.config = config or ModelConfig()
        self.models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.monitor_task = None
        
    async def initialize(self):
        """初始化管理器"""
        if self.running:
            return
            
        self.running = True
        print("正在初始化模型管理器...")
        
        # 加载模型配置
        await self._load_model_configs()
        
        # 初始化监控任务
        self.monitor_task = asyncio.create_task(self._monitor_status())
        
        print("模型管理器初始化完成")
        
    async def shutdown(self):
        """关闭管理器"""
        if not self.running:
            return
            
        self.running = False
        print("正在关闭模型管理器...")
        
        # 取消监控任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # 清理所有模型
        for model in self.models.values():
            model.cleanup()
            
        self.models.clear()
        print("模型管理器已关闭")
        
    async def load_model(self, model_name: str, model_class: Type[BaseModel]) -> Optional[BaseModel]:
        """加载模型"""
        try:
            if model_name in self.models:
                return self.models[model_name]
                
            # 获取模型配置
            model_config = self.model_configs.get(model_name)
            if not model_config:
                print(f"未找到模型配置: {model_name}")
                return None
                
            # 创建模型实例
            model = model_class()
            
            # 初始化模型
            if not await model.initialize():
                print(f"模型初始化失败: {model_name}")
                return None
                
            # 预热模型
            await model.warmup()
            
            # 保存模型实例
            self.models[model_name] = model
            
            return model
            
        except Exception as e:
            print(f"加载模型出错 {model_name}: {e}")
            return None
            
    async def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.models:
            model = self.models[model_name]
            model.cleanup()
            del self.models[model_name]
            
    async def get_model(self, model_name: str) -> Optional[BaseModel]:
        """获取模型实例"""
        return self.models.get(model_name)
        
    async def predict(self, model_name: str, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """模型推理"""
        try:
            model = await self.get_model(model_name)
            if not model:
                print(f"模型未加载: {model_name}")
                return None
                
            # 验证输入
            if not model.validate_input(inputs):
                print(f"输入数据验证失败: {model_name}")
                return None
                
            # 执行推理
            outputs = await model.predict(inputs)
            
            # 验证输出
            if not model.validate_output(outputs):
                print(f"输出数据验证失败: {model_name}")
                return None
                
            return outputs
            
        except Exception as e:
            print(f"模型推理出错 {model_name}: {e}")
            return None
            
    async def _load_model_configs(self):
        """加载模型配置"""
        try:
            # 检查配置目录
            if not os.path.exists(self.config.config_dir):
                print(f"配置目录不存在: {self.config.config_dir}")
                return
                
            # 加载所有配置文件
            for filename in os.listdir(self.config.config_dir):
                if filename.endswith('.json'):
                    model_name = filename[:-5]  # 移除.json后缀
                    config_path = os.path.join(self.config.config_dir, filename)
                    
                    # 读取配置
                    with open(config_path, 'r') as f:
                        self.model_configs[model_name] = eval(f.read())
                        
        except Exception as e:
            print(f"加载模型配置出错: {e}")
            
    async def _monitor_status(self):
        """监控状态"""
        while self.running:
            try:
                # 检查每个模型的状态
                for name, model in self.models.items():
                    # 检查GPU内存
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                        if memory_allocated > self.config.max_gpu_memory:
                            print(f"警告: GPU内存使用过高 ({memory_allocated:.1f}MB)")
                            
                    # 检查模型大小
                    model_size = model.get_model_size() / 1024**2  # MB
                    if model_size > self.config.max_model_size:
                        print(f"警告: 模型{name}大小过大 ({model_size:.1f}MB)")
                        
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                print(f"监控状态出错: {e}")
                await asyncio.sleep(5)
                
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        status = {
            'running': self.running,
            'loaded_models': {},
            'gpu_memory': None
        }
        
        # 获取已加载模型的信息
        for name, model in self.models.items():
            try:
                metadata = model.get_metadata()
                status['loaded_models'][name] = {
                    'metadata': metadata.__dict__,
                    'size': model.get_model_size() / 1024**2  # MB
                }
            except Exception as e:
                status['loaded_models'][name] = {'error': str(e)}
                
        # 获取GPU信息
        if torch.cuda.is_available():
            status['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2  # MB
            }
            
        return status
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.shutdown() 
"""
LLaVA处理器模块 - 处理图像分析请求
"""
import torch
from PIL import Image
import logging
from typing import Dict, Optional
from transformers import PreTrainedModel

from ..utils.logger import setup_logger
from config.sbs_prompt import SBS_PROMPT

logger = setup_logger('llava_processor')

class LLaVAProcessor:
    """LLaVA处理器类
    
    该类负责处理图像分析请求，使用LLaVA模型生成分析结果。
    
    Attributes:
        model (PreTrainedModel): LLaVA模型实例
        device (torch.device): 运行设备
        max_new_tokens (int): 生成时的最大新token数
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        max_new_tokens: int = 1000
    ):
        """初始化LLaVA处理器
        
        Args:
            model: LLaVA模型实例
            max_new_tokens: 生成时的最大新token数
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.max_new_tokens = max_new_tokens
        
    async def process_image(self, image: Image.Image) -> Dict:
        """处理图像并生成分析结果
        
        Args:
            image: PIL图像对象
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 准备输入
            inputs = self.prepare_inputs(image)
            
            # 生成分析结果
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
            # 解码输出
            response = self.model.decode(outputs[0])
            
            # 解析JSON响应
            try:
                import json
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败: {response}")
                return {
                    'success': False,
                    'error': '响应格式错误'
                }
                
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def prepare_inputs(self, image: Image.Image) -> Dict:
        """准备模型输入
        
        Args:
            image: PIL图像对象
            
        Returns:
            Dict: 模型输入
        """
        # 调整图像大小
        image = self.resize_image(image)
        
        # 转换为张量
        image_tensor = self.image_to_tensor(image)
        
        # 添加提示
        inputs = {
            'input_ids': self.tokenize(SBS_PROMPT),
            'pixel_values': image_tensor
        }
        
        return inputs
        
    def resize_image(self, image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
        """调整图像大小
        
        Args:
            image: PIL图像对象
            size: 目标大小
            
        Returns:
            Image.Image: 调整后的图像
        """
        return image.resize(size, Image.LANCZOS)
        
    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量
        
        Args:
            image: PIL图像对象
            
        Returns:
            torch.Tensor: 图像张量
        """
        # 转换为numpy数组
        import numpy as np
        image_array = np.array(image)
        
        # 标准化
        image_array = image_array / 255.0
        image_array = (image_array - 0.5) / 0.5
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_array)
        
        # 调整维度
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        
        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
        
    def tokenize(self, text: str) -> torch.Tensor:
        """对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: token ID张量
        """
        tokens = self.model.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        return tokens['input_ids'].to(self.device) 
"""
SBS分析器模型 - 基于LLaVA的金融图表分析
"""
import os
import torch
from PIL import Image
import logging
import re
from typing import Dict, Optional, Union, List
from pathlib import Path
import traceback
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

from ..utils.logger import setup_logger
from config.config import SBS_PROMPT, SBS_ANNOTATION_CONFIG
from .llava_processor import LLaVAProcessor
from ..analysis.simplified_confidence import SimplifiedConfidenceCalculator
from ..visualization.result_formatter import ResultFormatter
from ..visualization.annotator import ChartAnnotator

logger = setup_logger('sbs_analyzer')

class SBSAnalyzer:
    """
    SBS分析器类 - 使用LLaVA模型进行金融图表分析
    
    该类实现了基于LLaVA（Large Language and Vision Assistant）的金融图表分析功能，
    专门用于识别和分析SBS（Sequence Based Signal）交易序列。
    
    Attributes:
        device (torch.device): 运行设备
        model (PreTrainedModel): LLaVA模型实例
        processor (ProcessorMixin): LLaVA处理器实例
        model_path (str): 模型路径
        max_new_tokens (int): 生成时的最大新token数
        prompt_template (str): 分析提示模板
    """
    
    def __init__(
        self,
        base_model: str,
        device: str = "cuda",
        max_new_tokens: int = 1000,
        prompt_template: Optional[str] = None
    ):
        """初始化SBS分析器
        
        Args:
            base_model (str): LLaVA模型路径
            device (str): 运行设备 ("cuda" 或 "cpu")
            max_new_tokens (int): 生成时的最大新token数
            prompt_template (str, optional): 自定义提示模板
        
        Raises:
            RuntimeError: 模型加载失败时抛出
            ValueError: 参数验证失败时抛出
        """
        try:
            # 验证和设置设备
            self.device = self._setup_device(device)
            self.model_path = base_model
            self.max_new_tokens = max_new_tokens
            
            # 加载模型和处理器
            logger.info(f"正在从 {base_model} 加载LLaVA模型...")
            self.model, self.processor = self._load_model_and_processor()
            
            # 设置提示模板和配置
            self.prompt = SBS_PROMPT
            self.annotation_config = SBS_ANNOTATION_CONFIG
            
            # 初始化组件
            self.llava_processor = LLaVAProcessor(self.model)
            self.confidence_calculator = SimplifiedConfidenceCalculator()
            self.result_formatter = ResultFormatter()
            self.chart_annotator = ChartAnnotator({
                'min_price': 0,  # 这些值会在分析时动态更新
                'max_price': 100
            })
            
            logger.info(f"SBS分析器初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            error_msg = f"SBS分析器初始化失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    async def process_image(self, image_path: str) -> Dict:
        """处理图片并生成分析结果
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            Dict: 分析结果
            
        Raises:
            FileNotFoundError: 图片文件不存在时抛出
            RuntimeError: 处理过程出错时抛出
        """
        try:
            # 验证图片文件
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
                
            # 加载和处理图片
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 使用LLaVA处理器进行基础分析
            llava_output = await self.llava_processor.process_image(image)
            
            # 计算置信度
            confidence = self.confidence_calculator.calculate_confidence(llava_output)
            
            # 格式化结果
            formatted_result = self.result_formatter.format_json(llava_output, confidence)
            
            # 标注图表
            annotated_image_path = await self.chart_annotator.annotate_signal(
                image_path, 
                formatted_result
            )
            
            # 添加标注后的图片路径到结果中
            formatted_result['annotated_image'] = annotated_image_path
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"图片处理失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    def _setup_device(self, device: str) -> torch.device:
        """设置运行设备
        
        Args:
            device (str): 设备名称
            
        Returns:
            torch.device: 设备实例
            
        Raises:
            ValueError: 设备不可用时抛出
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU")
            device = "cpu"
        return torch.device(device)
        
    def _load_model_and_processor(self) -> tuple[PreTrainedModel, ProcessorMixin]:
        """加载模型和处理器
        
        Returns:
            tuple: (model, processor)
            
        Raises:
            RuntimeError: 加载失败时抛出
        """
        try:
            # 加载配置
            config = AutoConfig.from_pretrained(self.model_path)
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # 加载处理器
            processor = AutoProcessor.from_pretrained(self.model_path)
            
            # 移动模型到指定设备
            if self.device.type == "cpu":
                model = model.to(self.device)
                
            return model, processor
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
            
    async def warmup(self) -> bool:
        """模型预热
        
        执行一次空的推理以预热模型
        
        Returns:
            bool: 预热是否成功
        """
        try:
            # 创建一个1x3x224x224的随机张量作为测试输入
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            
            # 执行一次前向传播
            with torch.inference_mode():
                _ = self.model(dummy_image)
                
            logger.info("模型预热完成")
            return True
            
        except Exception as e:
            logger.error(f"模型预热失败: {e}")
            return False
            
    def __del__(self):
        """析构函数
        
        清理GPU内存
        """
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"清理资源失败: {e}") 
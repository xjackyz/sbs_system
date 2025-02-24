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
            
            # 设置生成参数
            self.prompt_template = prompt_template
            
            logger.info(f"SBS分析器初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            error_msg = f"SBS分析器初始化失败: {str(e)}\n{traceback.format_exc()}"
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
                
            # 准备输入
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)
            
            # 添加提示模板
            if self.prompt_template:
                inputs['prompt'] = self.prompt_template
                
            # 生成分析结果
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
                
            # 解码输出
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 解析结果
            return self._parse_response(response)
            
        except Exception as e:
            error_msg = f"图片处理失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    def _parse_response(self, response: str) -> Dict:
        """解析模型输出的文本响应
        
        Args:
            response (str): 模型输出的文本
            
        Returns:
            Dict: 解析后的结构化数据
        """
        try:
            # 初始化结果字典
            result = {
                'sequence_evaluation': {
                    'validity': False,
                    'completeness': 0,
                    'confidence': 0
                },
                'key_points': {
                    'breakout': None,
                    'point1': None,
                    'point2': None,
                    'point3': None,
                    'point4': None
                },
                'trading_signal': {
                    'direction': None,
                    'entry_zone': {'min': None, 'max': None},
                    'stop_loss': None,
                    'target': None
                },
                'trend_analysis': {
                    'sma20_trend': None,
                    'sma200_trend': None,
                    'overall_trend': None
                },
                'risk_assessment': {
                    'risk_level': None,
                    'main_risks': None
                }
            }
            
            # 使用更严格的正则表达式进行解析
            try:
                # 解析序列评估
                sequence_match = re.search(
                    r"序列评估：.*?有效性：\s*\[([是否])\].*?完整度：\s*\[(\d+)%?\].*?可信度：\s*\[(\d+)%?\]",
                    response,
                    re.DOTALL
                )
                if sequence_match:
                    result['sequence_evaluation'].update({
                        'validity': sequence_match.group(1) == '是',
                        'completeness': int(sequence_match.group(2)),
                        'confidence': int(sequence_match.group(3))
                    })
            except Exception as e:
                logger.warning(f"序列评估解析失败: {e}")
                
            try:
                # 解析关键点位
                points_match = re.search(
                    r"关键点位：.*?突破点：\s*\[([^\]]+)\].*?Point 1：\s*\[([^\]]+)\].*?Point 2：\s*\[([^\]]+)\].*?Point 3：\s*\[([^\]]+)\].*?Point 4：\s*\[([^\]]+)\]",
                    response,
                    re.DOTALL
                )
                if points_match:
                    result['key_points'].update({
                        'breakout': self._parse_price(points_match.group(1)),
                        'point1': self._parse_price(points_match.group(2)),
                        'point2': self._parse_price(points_match.group(3)),
                        'point3': self._parse_price(points_match.group(4)),
                        'point4': self._parse_price(points_match.group(5))
                    })
            except Exception as e:
                logger.warning(f"关键点位解析失败: {e}")
                
            try:
                # 解析交易信号
                signal_match = re.search(
                    r"交易信号：.*?方向：\s*\[([^\]]+)\].*?入场区域：\s*\[([^\]]+)\].*?止损位：\s*\[([^\]]+)\].*?目标位：\s*\[([^\]]+)\]",
                    response,
                    re.DOTALL
                )
                if signal_match:
                    result['trading_signal'].update({
                        'direction': signal_match.group(1),
                        'entry_zone': self._parse_price_range(signal_match.group(2)),
                        'stop_loss': self._parse_price(signal_match.group(3)),
                        'target': self._parse_price(signal_match.group(4))
                    })
            except Exception as e:
                logger.warning(f"交易信号解析失败: {e}")
                
            try:
                # 解析趋势分析
                trend_match = re.search(
                    r"趋势辅助分析：.*?SMA20趋势：\s*\[([^\]]+)\].*?SMA200趋势：\s*\[([^\]]+)\].*?整体趋势评估：\s*\[([^\]]+)\]",
                    response,
                    re.DOTALL
                )
                if trend_match:
                    result['trend_analysis'].update({
                        'sma20_trend': trend_match.group(1),
                        'sma200_trend': trend_match.group(2),
                        'overall_trend': trend_match.group(3)
                    })
            except Exception as e:
                logger.warning(f"趋势分析解析失败: {e}")
                
            try:
                # 解析风险评估
                risk_match = re.search(
                    r"风险评估：.*?风险等级：\s*\[([^\]]+)\].*?主要风险点：\s*\[([^\]]+)\]",
                    response,
                    re.DOTALL
                )
                if risk_match:
                    result['risk_assessment'].update({
                        'risk_level': risk_match.group(1),
                        'main_risks': risk_match.group(2)
                    })
            except Exception as e:
                logger.warning(f"风险评估解析失败: {e}")
                
            # 添加原始响应
            result['raw_response'] = response
            
            return result
            
        except Exception as e:
            logger.error(f"响应解析失败: {e}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'raw_response': response
            }
            
    def _parse_price(self, price_str: str) -> Optional[float]:
        """解析价格字符串
        
        Args:
            price_str (str): 价格字符串
            
        Returns:
            Optional[float]: 解析后的价格，如果解析失败则返回None
        """
        try:
            # 移除所有空白字符
            price_str = re.sub(r'\s+', '', price_str)
            
            # 如果是"未知"或空字符串，返回None
            if price_str in ['未知', '']:
                return None
                
            # 尝试将字符串转换为浮点数
            return float(price_str)
            
        except ValueError:
            logger.warning(f"价格解析失败: {price_str}")
            return None
            
    def _parse_price_range(self, range_str: str) -> Dict[str, Optional[float]]:
        """解析价格范围字符串
        
        Args:
            range_str (str): 价格范围字符串
            
        Returns:
            Dict[str, Optional[float]]: 包含最小值和最大值的字典
        """
        try:
            # 移除所有空白字符
            range_str = re.sub(r'\s+', '', range_str)
            
            # 如果是"未知"或空字符串，返回None
            if range_str in ['未知', '']:
                return {'min': None, 'max': None}
                
            # 尝试匹配范围格式（例如：1234.5678-2345.6789）
            range_match = re.search(r'([\d.]+)[-~到至]([\d.]+)', range_str)
            if range_match:
                return {
                    'min': float(range_match.group(1)),
                    'max': float(range_match.group(2))
                }
                
            # 如果是单个数字，则最小值和最大值相同
            return {
                'min': float(range_str),
                'max': float(range_str)
            }
            
        except ValueError:
            logger.warning(f"价格范围解析失败: {range_str}")
            return {'min': None, 'max': None}
            
    @staticmethod
    def format_price(price: Optional[float]) -> str:
        """格式化价格
        
        Args:
            price (Optional[float]): 价格值
            
        Returns:
            str: 格式化后的价格字符串
        """
        if price is None:
            return "未知"
        return f"{price:.4f}"
        
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
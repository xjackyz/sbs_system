"""
LLaVA分析器 - 整合图表分析和标注功能
"""
import logging
from typing import Dict, Optional
from pathlib import Path
import traceback

from .llava_processor import LLaVAProcessor
from ..analysis.simplified_confidence import SimplifiedConfidenceCalculator
from ..visualization.result_formatter import ResultFormatter
from ..visualization.annotator import ChartAnnotator
from ..utils.logger import setup_logger
from config.config import SBS_PROMPT, SBS_ANNOTATION_CONFIG

logger = setup_logger('llava_analyzer')

class LLaVAAnalyzer:
    def __init__(self, config: Dict):
        """初始化LLaVA分析器
        
        Args:
            config: 配置字典
        """
        try:
            self.config = config
            
            # 初始化各个组件
            self.llava_processor = LLaVAProcessor(
                model=config.get('model'),
                prompt_template=SBS_PROMPT.get('output_format')
            )
            
            self.confidence_calculator = SimplifiedConfidenceCalculator()
            self.result_formatter = ResultFormatter()
            self.annotator = ChartAnnotator(config)
            
            logger.info("LLaVA分析器初始化完成")
            
        except Exception as e:
            error_msg = f"LLaVA分析器初始化失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def analyze_chart(self, image_path: str) -> Dict:
        """分析图表并在有效信号时进行标注
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 使用LLaVA处理器进行基础分析
            llava_output = await self.llava_processor.process_image(image_path)
            
            # 计算置信度
            confidence = self.confidence_calculator.calculate_confidence(llava_output)
            
            # 格式化结果
            analysis_result = self.result_formatter.format_json(llava_output, confidence)
            analysis_result['success'] = True
            
            # 如果是有效信号，进行标注
            if (analysis_result.get('sequence_evaluation', {}).get('validity') == '是' and
                analysis_result.get('confidence', 0) >= 70):
                try:
                    # 标注图表
                    annotated_image_path = await self.annotator.annotate_signal(
                        image_path,
                        analysis_result
                    )
                    analysis_result['annotated_image'] = annotated_image_path
                except Exception as e:
                    logger.warning(f"图表标注失败: {e}")
                    analysis_result['annotation_error'] = str(e)
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"图表分析失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
    async def _analyze_with_llava(self, image_path: str) -> Dict:
        """使用LLaVA模型进行图表分析
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 使用LLaVA处理器进行分析
            llava_output = await self.llava_processor.process_image(image_path)
            
            # 计算置信度
            confidence = self.confidence_calculator.calculate_confidence(llava_output)
            
            # 格式化结果
            result = self.result_formatter.format_json(llava_output, confidence)
            result['success'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"LLaVA分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _validate_signal(self, analysis_result: Dict) -> bool:
        """验证是否是有效信号
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            bool: 是否有效
        """
        try:
            sequence_eval = analysis_result.get('sequence_evaluation', {})
            return (sequence_eval.get('validity') == '是' and
                   sequence_eval.get('confidence', 0) >= 70)
        except Exception as e:
            logger.error(f"信号验证失败: {e}")
            return False 
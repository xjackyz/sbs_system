"""
简化版置信度计算器
"""
from typing import Dict
import logging
from ..utils.logger import setup_logger

logger = setup_logger('confidence_calculator')

class SimplifiedConfidenceCalculator:
    def __init__(self):
        """初始化置信度计算器"""
        self.weights = {
            'point_visibility': 0.4,    # 关键点是否清晰可见
            'pattern_standard': 0.4,    # 形态是否符合标准
            'sequence_complete': 0.2    # 序列是否完整
        }
    
    def calculate_confidence(self, llava_output: Dict) -> float:
        """计算简化版置信度
        
        Args:
            llava_output: LLaVA模型的输出结果
            
        Returns:
            float: 计算得到的置信度 (0-1)
        """
        try:
            scores = {
                'point_visibility': self._check_points_visibility(llava_output),
                'pattern_standard': self._check_pattern_standard(llava_output),
                'sequence_complete': self._check_sequence_complete(llava_output)
            }
            
            confidence = sum(scores[k] * self.weights[k] for k in self.weights)
            logger.debug(f"置信度计算结果: {confidence:.2f}, 详细得分: {scores}")
            
            return confidence
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.0
    
    def _check_points_visibility(self, output: Dict) -> float:
        """检查关键点的可见性
        
        Args:
            output: 分析输出
            
        Returns:
            float: 可见性得分 (0-1)
        """
        if not output.get('points'):
            return 0.0
            
        visible_points = sum(1 for point in output['points'] 
                           if point.get('visibility') == 'clear')
        return visible_points / 5  # 5个关键点
    
    def _check_pattern_standard(self, output: Dict) -> float:
        """检查形态标准程度
        
        Args:
            output: 分析输出
            
        Returns:
            float: 标准程度得分 (0-1)
        """
        if not output.get('points'):
            return 0.0
            
        standard_points = sum(1 for point in output['points'] 
                            if point.get('standard') == True)
        return standard_points / 5
    
    def _check_sequence_complete(self, output: Dict) -> float:
        """检查序列完整性
        
        Args:
            output: 分析输出
            
        Returns:
            float: 完整性得分 (0-1)
        """
        points = output.get('points', [])
        return 1.0 if len(points) == 5 else 0.0 
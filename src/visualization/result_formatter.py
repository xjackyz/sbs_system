"""
分析结果格式化工具
"""
from typing import Dict
import logging
from ..utils.logger import setup_logger

logger = setup_logger('result_formatter')

class ResultFormatter:
    def format_analysis(self, 
                       llava_output: Dict, 
                       confidence: float) -> str:
        """格式化分析结果
        
        Args:
            llava_output: LLaVA模型的输出结果
            confidence: 计算得到的置信度
            
        Returns:
            str: 格式化后的分析结果
        """
        try:
            result = "SBS序列分析结果:\n\n"
            
            # 添加识别到的关键点
            result += "识别到的关键点:\n"
            for point in llava_output.get('points', []):
                result += f"- {point.get('name', '未知点位')}: "
                result += "清晰可见" if point.get('visibility') == 'clear' else "不够清晰"
                result += "\n"
            
            # 添加置信度
            result += f"\n整体置信度: {confidence:.1%}\n"
            
            # 添加建议
            result += "\n建议操作: "
            if confidence >= 0.8:
                result += "高置信度信号，建议按计划交易"
            elif confidence >= 0.6:
                result += "中等置信度，建议谨慎操作"
            else:
                result += "低置信度，建议等待更好的机会"
                
            # 添加风险提示
            result += "\n\n风险提示: "
            if confidence < 0.6:
                result += "当前形态不够标准，建议观察更多确认信号"
            elif any(point.get('visibility') != 'clear' 
                    for point in llava_output.get('points', [])):
                result += "部分关键点位不够清晰，请注意控制风险"
            else:
                result += "风险可控，请严格执行交易计划"
                
            return result
            
        except Exception as e:
            logger.error(f"结果格式化失败: {e}")
            return "结果格式化失败，请检查日志"
            
    def format_json(self, 
                   llava_output: Dict, 
                   confidence: float) -> Dict:
        """将分析结果格式化为JSON格式
        
        Args:
            llava_output: LLaVA模型的输出结果
            confidence: 计算得到的置信度
            
        Returns:
            Dict: JSON格式的分析结果
        """
        try:
            return {
                'points': llava_output.get('points', []),
                'confidence': confidence,
                'recommendation': self._get_recommendation(confidence),
                'risk_level': self._get_risk_level(confidence, llava_output)
            }
        except Exception as e:
            logger.error(f"JSON格式化失败: {e}")
            return {
                'error': str(e)
            }
            
    def _get_recommendation(self, confidence: float) -> str:
        """根据置信度获取建议
        
        Args:
            confidence: 置信度
            
        Returns:
            str: 建议操作
        """
        if confidence >= 0.8:
            return "高置信度信号，建议按计划交易"
        elif confidence >= 0.6:
            return "中等置信度，建议谨慎操作"
        else:
            return "低置信度，建议等待更好的机会"
            
    def _get_risk_level(self, 
                       confidence: float, 
                       llava_output: Dict) -> str:
        """获取风险等级
        
        Args:
            confidence: 置信度
            llava_output: LLaVA模型的输出结果
            
        Returns:
            str: 风险等级
        """
        if confidence < 0.6:
            return "高风险"
        elif any(point.get('visibility') != 'clear' 
                for point in llava_output.get('points', [])):
            return "中等风险"
        else:
            return "低风险" 
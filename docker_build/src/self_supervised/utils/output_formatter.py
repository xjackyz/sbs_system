import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OutputRequirements:
    """输出要求配置"""
    sequence_validity_threshold: float = 0.8
    confidence_threshold: float = 0.7
    required_points: int = 5

class OutputFormatter:
    """输出格式化器"""
    
    def __init__(self, requirements: Optional[OutputRequirements] = None):
        """
        初始化格式化器
        
        Args:
            requirements: 输出要求配置
        """
        self.requirements = requirements or OutputRequirements()
        
    def format_model_outputs(self, raw_outputs: Dict) -> Dict:
        """
        格式化模型输出
        
        Args:
            raw_outputs: 原始模型输出
            
        Returns:
            格式化后的输出
        """
        try:
            formatted = {
                'sequence_validity': {
                    'score': float(raw_outputs.get('validity_score', 0)),
                    'is_valid': False,
                    'timestamp': datetime.now().isoformat()
                },
                'key_points': [],
                'trading_signals': {
                    'action': 'hold',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # 验证序列有效性
            if formatted['sequence_validity']['score'] >= self.requirements.sequence_validity_threshold:
                formatted['sequence_validity']['is_valid'] = True
            
            # 格式化关键点
            if 'points' in raw_outputs:
                formatted['key_points'] = self._format_key_points(raw_outputs['points'])
            
            # 格式化交易信号
            if 'signals' in raw_outputs:
                formatted['trading_signals'] = self._format_trading_signals(raw_outputs['signals'])
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化模型输出失败: {e}")
            return self._get_default_output()
            
    def format_evaluation_report(self, evaluation_results: Dict) -> Dict:
        """
        格式化评估报告
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            格式化后的评估报告
        """
        try:
            formatted = {
                'performance_metrics': {
                    'accuracy': float(evaluation_results.get('accuracy', 0)),
                    'precision': float(evaluation_results.get('precision', 0)),
                    'recall': float(evaluation_results.get('recall', 0))
                },
                'error_analysis': {
                    'false_positives': int(evaluation_results.get('false_positives', 0)),
                    'false_negatives': int(evaluation_results.get('false_negatives', 0)),
                    'confusion_matrix': evaluation_results.get('confusion_matrix', np.zeros((2, 2)))
                }
            }
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化评估报告失败: {e}")
            return self._get_default_evaluation_report()
    
    def _format_key_points(self, points: List[Dict]) -> List[Dict]:
        """格式化关键点数据"""
        formatted_points = []
        
        for point in points:
            if point.get('confidence', 0) >= self.requirements.confidence_threshold:
                formatted_points.append({
                    'index': int(point.get('index', 0)),
                    'price': float(point.get('price', 0)),
                    'timestamp': point.get('timestamp', datetime.now().isoformat()),
                    'confidence': float(point.get('confidence', 0))
                })
        
        return formatted_points
    
    def _format_trading_signals(self, signals: Dict) -> Dict:
        """格式化交易信号"""
        action = signals.get('action', 'hold')
        confidence = float(signals.get('confidence', 0))
        
        # 如果置信度低于阈值，转换为hold
        if confidence < self.requirements.confidence_threshold:
            action = 'hold'
            confidence = 0.0
            
        return {
            'action': action,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_output(self) -> Dict:
        """获取默认输出格式"""
        return {
            'sequence_validity': {
                'score': 0.0,
                'is_valid': False,
                'timestamp': datetime.now().isoformat()
            },
            'key_points': [],
            'trading_signals': {
                'action': 'hold',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_default_evaluation_report(self) -> Dict:
        """获取默认评估报告格式"""
        return {
            'performance_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0
            },
            'error_analysis': {
                'false_positives': 0,
                'false_negatives': 0,
                'confusion_matrix': np.zeros((2, 2))
            }
        } 
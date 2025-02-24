import numpy as np
from typing import Dict, List, Tuple
import torch
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """评估指标配置"""
    point_distance_threshold: int = 5    # 关键点位置容差
    price_error_threshold: float = 0.002 # 价格误差阈值
    min_confidence_threshold: float = 0.7 # 最小置信度阈值

class SequenceMetrics:
    """序列评估指标计算器"""
    
    def __init__(self, config: MetricsConfig = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or MetricsConfig()
        
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: 模型预测结果列表
            targets: 目标值列表
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 1. 序列识别准确率
        metrics['sequence_accuracy'] = self._calculate_sequence_accuracy(predictions, targets)
        
        # 2. 关键点位置精确度
        point_precision = self._calculate_point_precision(predictions, targets)
        metrics['point_precision'] = point_precision
        
        # 3. 关键点位置召回率
        point_recall = self._calculate_point_recall(predictions, targets)
        metrics['point_recall'] = point_recall
        
        # 4. 序列完整性评分
        completion_score = self._calculate_completion_score(predictions, targets)
        metrics['completion_score'] = completion_score
        
        # 5. F1分数
        if point_precision + point_recall > 0:
            metrics['f1_score'] = 2 * (point_precision * point_recall) / (point_precision + point_recall)
        else:
            metrics['f1_score'] = 0.0
            
        return metrics
        
    def _calculate_sequence_accuracy(self, predictions: List[Dict], targets: List[Dict]) -> float:
        """计算序列识别准确率"""
        try:
            correct = 0
            total = len(predictions)
            
            for pred, target in zip(predictions, targets):
                # 检查序列类型是否匹配
                if pred['sequence_type'] == target['sequence_type']:
                    # 检查关键点数量是否匹配
                    pred_points = len([p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold])
                    target_points = len(target['points'])
                    
                    if pred_points == target_points:
                        correct += 1
                        
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"序列准确率计算失败: {e}")
            return 0.0
            
    def _calculate_point_precision(self, predictions: List[Dict], targets: List[Dict]) -> float:
        """计算关键点位置精确度"""
        try:
            total_correct = 0
            total_predicted = 0
            
            for pred, target in zip(predictions, targets):
                pred_points = [p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold]
                target_points = target['points']
                
                for pred_point in pred_points:
                    # 检查是否有匹配的目标点
                    if self._find_matching_point(pred_point, target_points):
                        total_correct += 1
                        
                total_predicted += len(pred_points)
                
            return total_correct / total_predicted if total_predicted > 0 else 0.0
            
        except Exception as e:
            logger.error(f"点位精确度计算失败: {e}")
            return 0.0
            
    def _calculate_point_recall(self, predictions: List[Dict], targets: List[Dict]) -> float:
        """计算关键点位置召回率"""
        try:
            total_correct = 0
            total_target = 0
            
            for pred, target in zip(predictions, targets):
                pred_points = [p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold]
                target_points = target['points']
                
                for target_point in target_points:
                    # 检查是否被预测到
                    if self._find_matching_point(target_point, pred_points):
                        total_correct += 1
                        
                total_target += len(target_points)
                
            return total_correct / total_target if total_target > 0 else 0.0
            
        except Exception as e:
            logger.error(f"点位召回率计算失败: {e}")
            return 0.0
            
    def _calculate_completion_score(self, predictions: List[Dict], targets: List[Dict]) -> float:
        """计算序列完整性评分"""
        try:
            total_score = 0.0
            total_sequences = len(predictions)
            
            for pred, target in zip(predictions, targets):
                # 计算点位匹配得分
                point_score = self._calculate_point_matching_score(pred, target)
                
                # 计算价格准确性得分
                price_score = self._calculate_price_accuracy_score(pred, target)
                
                # 计算时序正确性得分
                sequence_score = self._calculate_temporal_order_score(pred, target)
                
                # 综合评分
                sequence_completion = (point_score + price_score + sequence_score) / 3
                total_score += sequence_completion
                
            return total_score / total_sequences if total_sequences > 0 else 0.0
            
        except Exception as e:
            logger.error(f"完整性评分计算失败: {e}")
            return 0.0
            
    def _find_matching_point(self, point: Dict, point_list: List[Dict]) -> bool:
        """查找匹配的点位"""
        for p in point_list:
            # 检查点位类型
            if p['point_type'] != point['point_type']:
                continue
                
            # 检查位置距离
            if abs(p['index'] - point['index']) > self.config.point_distance_threshold:
                continue
                
            # 检查价格误差
            price_error = abs(p['price'] - point['price']) / point['price']
            if price_error > self.config.price_error_threshold:
                continue
                
            return True
            
        return False
        
    def _calculate_point_matching_score(self, pred: Dict, target: Dict) -> float:
        """计算点位匹配得分"""
        try:
            pred_points = [p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold]
            target_points = target['points']
            
            if not target_points:
                return 1.0 if not pred_points else 0.0
                
            matched_points = 0
            for target_point in target_points:
                if self._find_matching_point(target_point, pred_points):
                    matched_points += 1
                    
            return matched_points / len(target_points)
            
        except Exception as e:
            logger.error(f"点位匹配得分计算失败: {e}")
            return 0.0
            
    def _calculate_price_accuracy_score(self, pred: Dict, target: Dict) -> float:
        """计算价格准确性得分"""
        try:
            pred_points = [p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold]
            target_points = target['points']
            
            if not target_points or not pred_points:
                return 0.0
                
            total_error = 0.0
            matched_points = 0
            
            for target_point in target_points:
                for pred_point in pred_points:
                    if pred_point['point_type'] == target_point['point_type']:
                        error = abs(pred_point['price'] - target_point['price']) / target_point['price']
                        total_error += error
                        matched_points += 1
                        break
                        
            if matched_points == 0:
                return 0.0
                
            avg_error = total_error / matched_points
            return max(0.0, 1.0 - avg_error / self.config.price_error_threshold)
            
        except Exception as e:
            logger.error(f"价格准确性得分计算失败: {e}")
            return 0.0
            
    def _calculate_temporal_order_score(self, pred: Dict, target: Dict) -> float:
        """计算时序正确性得分"""
        try:
            pred_points = [p for p in pred['points'] if p['confidence'] >= self.config.min_confidence_threshold]
            target_points = target['points']
            
            if not target_points or not pred_points:
                return 0.0
                
            # 检查点位顺序是否正确
            correct_order = 0
            total_pairs = 0
            
            for i in range(len(target_points) - 1):
                target_curr = target_points[i]
                target_next = target_points[i + 1]
                
                # 在预测点中找到对应的点
                pred_curr = None
                pred_next = None
                
                for p in pred_points:
                    if p['point_type'] == target_curr['point_type']:
                        pred_curr = p
                    elif p['point_type'] == target_next['point_type']:
                        pred_next = p
                        
                if pred_curr and pred_next:
                    total_pairs += 1
                    if pred_curr['index'] < pred_next['index']:
                        correct_order += 1
                        
            return correct_order / total_pairs if total_pairs > 0 else 0.0
            
        except Exception as e:
            logger.error(f"时序正确性得分计算失败: {e}")
            return 0.0 
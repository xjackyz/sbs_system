from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple

@dataclass
class ValidationConfig:
    """验证配置"""
    min_confidence_threshold: float = 0.75
    min_pattern_similarity: float = 0.85
    max_volatility_threshold: float = 0.03
    min_trend_strength: float = 0.6
    min_liquidity_score: float = 0.7
    validation_window: int = 100

class SequenceValidator:
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_history = []
        
    def validate_sequence(self, sequence_data: pd.DataFrame, 
                         prediction: Dict) -> ValidationResult:
        """多层验证序列"""
        try:
            # 1. 基础验证
            base_validation = self._validate_basic_requirements(sequence_data, prediction)
            if not base_validation['is_valid']:
                return self._create_validation_result(False, base_validation)

            # 2. 市场环境验证
            market_validation = self._validate_market_conditions(sequence_data)
            if not market_validation['is_valid']:
                return self._create_validation_result(False, market_validation)

            # 3. 模式质量验证
            pattern_validation = self._validate_pattern_quality(sequence_data, prediction)
            if not pattern_validation['is_valid']:
                return self._create_validation_result(False, pattern_validation)

            # 4. 一致性验证
            consistency_validation = self._validate_consistency(prediction)
            if not consistency_validation['is_valid']:
                return self._create_validation_result(False, consistency_validation)

            # 计算综合得分
            final_score = self._calculate_composite_score([
                base_validation,
                market_validation,
                pattern_validation,
                consistency_validation
            ])

            # 更新验证历史
            self._update_validation_history(final_score)

            return self._create_validation_result(True, {
                'score': final_score,
                'details': {
                    'base_validation': base_validation,
                    'market_validation': market_validation,
                    'pattern_validation': pattern_validation,
                    'consistency_validation': consistency_validation
                }
            })

        except Exception as e:
            return self._create_validation_result(False, {'error': str(e)})

    def _validate_basic_requirements(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """验证基本要求"""
        return {
            'is_valid': True if (
                len(data) >= self.config.validation_window and
                prediction.get('confidence', 0) >= self.config.min_confidence_threshold
            ) else False,
            'score': prediction.get('confidence', 0),
            'details': {
                'data_length': len(data),
                'confidence': prediction.get('confidence', 0)
            }
        }

    def _validate_market_conditions(self, data: pd.DataFrame) -> Dict:
        """验证市场条件"""
        volatility = self._calculate_volatility(data)
        trend_strength = self._calculate_trend_strength(data)
        liquidity = self._calculate_liquidity(data)

        is_valid = (
            volatility <= self.config.max_volatility_threshold and
            trend_strength >= self.config.min_trend_strength and
            liquidity >= self.config.min_liquidity_score
        )

        return {
            'is_valid': is_valid,
            'score': np.mean([
                1 - (volatility / self.config.max_volatility_threshold),
                trend_strength / self.config.min_trend_strength,
                liquidity / self.config.min_liquidity_score
            ]),
            'details': {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'liquidity': liquidity
            }
        }

    def _validate_pattern_quality(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """验证模式质量"""
        pattern_similarity = self._calculate_pattern_similarity(prediction)
        structure_quality = self._analyze_pattern_structure(prediction)
        time_alignment = self._check_time_alignment(prediction)

        is_valid = (
            pattern_similarity >= self.config.min_pattern_similarity and
            structure_quality['is_valid'] and
            time_alignment['is_valid']
        )

        return {
            'is_valid': is_valid,
            'score': np.mean([
                pattern_similarity,
                structure_quality['score'],
                time_alignment['score']
            ]),
            'details': {
                'pattern_similarity': pattern_similarity,
                'structure_quality': structure_quality,
                'time_alignment': time_alignment
            }
        }

    def _validate_consistency(self, prediction: Dict) -> Dict:
        """验证预测一致性"""
        if not self.validation_history:
            return {'is_valid': True, 'score': 1.0}

        recent_predictions = self.validation_history[-5:]
        consistency_score = self._calculate_prediction_consistency(prediction, recent_predictions)

        return {
            'is_valid': consistency_score >= 0.7,
            'score': consistency_score,
            'details': {
                'historical_consistency': consistency_score,
                'recent_predictions_count': len(recent_predictions)
            }
        }

    def _calculate_composite_score(self, validations: List[Dict]) -> float:
        """计算综合得分"""
        weights = {
            'base_validation': 0.3,
            'market_validation': 0.25,
            'pattern_validation': 0.25,
            'consistency_validation': 0.2
        }

        weighted_scores = []
        for validation, (key, weight) in zip(validations, weights.items()):
            weighted_scores.append(validation['score'] * weight)

        return np.sum(weighted_scores)

    def _update_validation_history(self, result: Dict):
        """更新验证历史"""
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        # 保持历史记录在合理范围内
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]

    def _create_validation_result(self, is_valid: bool, details: Dict) -> ValidationResult:
        """创建验证结果"""
        return ValidationResult(
            is_valid=is_valid,
            score=details.get('score', 0.0),
            details=details,
            stage='validation',
            timestamp=datetime.now().isoformat()
        )

    # ... (保留其他现有方法) 
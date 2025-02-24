from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

from config.config import TRADE_SIGNALS

logger = logging.getLogger(__name__)

@dataclass
class TradeValidationResult:
    """交易验证结果"""
    is_valid: bool
    signal_type: str  # 'entry' 或 'exit'
    entry_type: Optional[str] = None  # 'immediate' 或 'pullback'
    confidence: float = 0.0
    details: Dict = None
    timestamp: str = None

class TradeValidator:
    """交易信号验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.signals_config = TRADE_SIGNALS
        self.validation_history = []
        
    def validate_entry(self, sequence_data: Dict, market_data: Dict) -> TradeValidationResult:
        """
        验证入场信号
        
        Args:
            sequence_data: 序列数据和分析结果
            market_data: 市场数据和指标
            
        Returns:
            交易验证结果
        """
        try:
            # 1. 验证主要条件
            primary_validation = self._validate_primary_conditions(sequence_data)
            if not primary_validation['is_valid']:
                return self._create_validation_result(False, 'entry', details=primary_validation)
            
            # 2. 确定入场类型
            entry_type, entry_validation = self._determine_entry_type(sequence_data, market_data)
            if not entry_validation['is_valid']:
                return self._create_validation_result(False, 'entry', details=entry_validation)
            
            # 3. 计算综合得分
            confidence = self._calculate_entry_confidence(
                primary_validation['score'],
                entry_validation['score']
            )
            
            # 4. 创建验证结果
            result = self._create_validation_result(
                is_valid=True,
                signal_type='entry',
                entry_type=entry_type,
                confidence=confidence,
                details={
                    'primary_validation': primary_validation,
                    'entry_validation': entry_validation
                }
            )
            
            # 5. 更新验证历史
            self._update_validation_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"入场信号验证失败: {e}")
            return self._create_validation_result(False, 'entry')
            
    def validate_exit(self, position_data: Dict, market_data: Dict) -> TradeValidationResult:
        """
        验证出场信号
        
        Args:
            position_data: 持仓数据
            market_data: 市场数据和指标
            
        Returns:
            交易验证结果
        """
        try:
            # 1. 检查止盈条件
            tp_validation = self._check_take_profit_conditions(position_data, market_data)
            if tp_validation['is_valid']:
                return self._create_validation_result(
                    True, 'exit',
                    confidence=tp_validation['confidence'],
                    details={'exit_type': 'take_profit', **tp_validation}
                )
            
            # 2. 检查止损条件
            sl_validation = self._check_stop_loss_conditions(position_data, market_data)
            if sl_validation['is_valid']:
                return self._create_validation_result(
                    True, 'exit',
                    confidence=sl_validation['confidence'],
                    details={'exit_type': 'stop_loss', **sl_validation}
                )
            
            # 3. 检查保本条件
            be_validation = self._check_breakeven_conditions(position_data, market_data)
            if be_validation['is_valid']:
                return self._create_validation_result(
                    True, 'exit',
                    confidence=be_validation['confidence'],
                    details={'exit_type': 'breakeven', **be_validation}
                )
            
            return self._create_validation_result(False, 'exit')
            
        except Exception as e:
            logger.error(f"出场信号验证失败: {e}")
            return self._create_validation_result(False, 'exit')
            
    def _validate_primary_conditions(self, sequence_data: Dict) -> Dict:
        """验证主要入场条件"""
        try:
            config = self.signals_config['entry_conditions']['primary']
            
            # 检查SBS完整性
            if config['sbs_completion'] and not sequence_data.get('sbs_complete', False):
                return {'is_valid': False, 'score': 0.0, 'reason': 'SBS序列不完整'}
            
            # 检查形态清晰度
            pattern_clarity = sequence_data.get('pattern_clarity', 0)
            if pattern_clarity < config['pattern_clarity']:
                return {
                    'is_valid': False,
                    'score': pattern_clarity,
                    'reason': '形态清晰度不足'
                }
            
            # 检查SCE确认
            if config['sce_confirmation'] and not sequence_data.get('sce_confirmed', False):
                return {'is_valid': False, 'score': 0.0, 'reason': '缺少SCE确认'}
            
            # 计算综合得分
            score = np.mean([
                1.0 if sequence_data.get('sbs_complete', False) else 0.0,
                sequence_data.get('pattern_clarity', 0),
                1.0 if sequence_data.get('sce_confirmed', False) else 0.0
            ])
            
            return {
                'is_valid': True,
                'score': score,
                'details': {
                    'sbs_complete': sequence_data.get('sbs_complete', False),
                    'pattern_clarity': pattern_clarity,
                    'sce_confirmed': sequence_data.get('sce_confirmed', False)
                }
            }
            
        except Exception as e:
            logger.error(f"验证主要条件失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'reason': str(e)}
            
    def _determine_entry_type(self, sequence_data: Dict, market_data: Dict) -> Tuple[str, Dict]:
        """确定入场类型"""
        try:
            # 检查即时入场条件
            immediate_validation = self._validate_immediate_entry(sequence_data, market_data)
            if immediate_validation['is_valid']:
                return 'immediate', immediate_validation
            
            # 检查回调入场条件
            pullback_validation = self._validate_pullback_entry(sequence_data, market_data)
            if pullback_validation['is_valid']:
                return 'pullback', pullback_validation
            
            return None, {'is_valid': False, 'score': 0.0, 'reason': '无有效入场类型'}
            
        except Exception as e:
            logger.error(f"确定入场类型失败: {e}")
            return None, {'is_valid': False, 'score': 0.0, 'reason': str(e)}
            
    def _validate_immediate_entry(self, sequence_data: Dict, market_data: Dict) -> Dict:
        """验证即时入场"""
        config = self.signals_config['entry_conditions']['entry_types']['immediate']
        
        # 检查突破确认
        if config['break_confirmation'] and not sequence_data.get('break_confirmed', False):
            return {'is_valid': False, 'score': 0.0, 'reason': '缺少突破确认'}
        
        # 检查形态质量
        pattern_quality = sequence_data.get('pattern_quality', 'low')
        if pattern_quality != config['pattern_quality']:
            return {
                'is_valid': False,
                'score': 0.5 if pattern_quality == 'medium' else 0.0,
                'reason': '形态质量不足'
            }
        
        return {
            'is_valid': True,
            'score': 1.0,
            'details': {
                'break_confirmed': sequence_data.get('break_confirmed', False),
                'pattern_quality': pattern_quality
            }
        }
        
    def _validate_pullback_entry(self, sequence_data: Dict, market_data: Dict) -> Dict:
        """验证回调入场"""
        config = self.signals_config['entry_conditions']['entry_types']['pullback']
        
        # 检查回调质量
        retracement_quality = sequence_data.get('retracement_quality', 'poor')
        if retracement_quality != config['retracement_quality']:
            return {
                'is_valid': False,
                'score': 0.5 if retracement_quality == 'fair' else 0.0,
                'reason': '回调质量不足'
            }
        
        # 检查入场价格
        entry_price_quality = sequence_data.get('entry_price_quality', 'suboptimal')
        if entry_price_quality != config['entry_price']:
            return {
                'is_valid': False,
                'score': 0.5 if entry_price_quality == 'good' else 0.0,
                'reason': '入场价格不理想'
            }
        
        return {
            'is_valid': True,
            'score': 1.0,
            'details': {
                'retracement_quality': retracement_quality,
                'entry_price_quality': entry_price_quality
            }
        }
        
    def _check_take_profit_conditions(self, position_data: Dict, market_data: Dict) -> Dict:
        """检查止盈条件"""
        config = self.signals_config['exit_rules']['take_profit']
        current_price = market_data.get('current_price')
        entry_price = position_data.get('entry_price')
        
        if not (current_price and entry_price):
            return {'is_valid': False, 'confidence': 0.0}
        
        price_change = (current_price - entry_price) / entry_price
        
        # 检查每个止盈目标
        for level in config['levels']:
            if price_change >= level['target']:
                return {
                    'is_valid': True,
                    'confidence': 1.0,
                    'target_level': level,
                    'price_change': price_change
                }
        
        return {'is_valid': False, 'confidence': 0.0}
        
    def _check_stop_loss_conditions(self, position_data: Dict, market_data: Dict) -> Dict:
        """检查止损条件"""
        config = self.signals_config['exit_rules']['stop_loss']['placement']
        current_price = market_data.get('current_price')
        stop_price = position_data.get('stop_loss')
        
        if not (current_price and stop_price):
            return {'is_valid': False, 'confidence': 0.0}
        
        # 检查是否触及止损
        if (position_data['direction'] == 'long' and current_price <= stop_price) or \
           (position_data['direction'] == 'short' and current_price >= stop_price):
            return {
                'is_valid': True,
                'confidence': 1.0,
                'stop_price': stop_price,
                'current_price': current_price
            }
        
        return {'is_valid': False, 'confidence': 0.0}
        
    def _check_breakeven_conditions(self, position_data: Dict, market_data: Dict) -> Dict:
        """检查保本条件"""
        config = self.signals_config['exit_rules']['stop_loss']['breakeven']
        current_price = market_data.get('current_price')
        entry_price = position_data.get('entry_price')
        
        if not (current_price and entry_price):
            return {'is_valid': False, 'confidence': 0.0}
        
        price_change = abs(current_price - entry_price) / entry_price
        
        # 检查是否满足移动止损条件
        if price_change >= config['condition']:
            return {
                'is_valid': True,
                'confidence': 1.0,
                'price_change': price_change,
                'breakeven_level': entry_price * (1 + config['buffer'])
            }
        
        return {'is_valid': False, 'confidence': 0.0}
        
    def _calculate_entry_confidence(self, primary_score: float, entry_score: float) -> float:
        """计算入场信号置信度"""
        # 主要条件权重0.6，入场类型权重0.4
        return 0.6 * primary_score + 0.4 * entry_score
        
    def _create_validation_result(self, is_valid: bool, signal_type: str,
                                entry_type: Optional[str] = None,
                                confidence: float = 0.0,
                                details: Dict = None) -> TradeValidationResult:
        """创建验证结果"""
        return TradeValidationResult(
            is_valid=is_valid,
            signal_type=signal_type,
            entry_type=entry_type,
            confidence=confidence,
            details=details or {},
            timestamp=datetime.now().isoformat()
        )
        
    def _update_validation_history(self, result: TradeValidationResult):
        """更新验证历史"""
        self.validation_history.append({
            'timestamp': result.timestamp,
            'result': {
                'is_valid': result.is_valid,
                'signal_type': result.signal_type,
                'entry_type': result.entry_type,
                'confidence': result.confidence
            }
        })
        
        # 保持历史记录在合理范围内
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:] 
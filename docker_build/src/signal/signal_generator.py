import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger('signal_generator')

@dataclass
class SignalConfig:
    """信号生成器配置"""
    min_confidence: float = 0.75
    max_risk_ratio: float = 0.02
    min_reward_ratio: float = 2.0
    entry_zone_size: float = 0.001
    max_stop_distance: float = 0.02
    min_volume: int = 1000
    validation_window: int = 100
    history_size: int = 1000
    patterns: Dict = None
    filters: Dict = None
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.patterns is None:
            self.patterns = {
                'trend_continuation': {
                    'min_length': 5,
                    'max_length': 20,
                    'min_slope': 0.001,
                    'min_r2': 0.7,
                    'volume_increase': 1.5
                },
                'trend_reversal': {
                    'min_length': 3,
                    'max_length': 10,
                    'min_angle': 30,
                    'min_volume_increase': 1.5,
                    'consolidation_bars': 5
                }
            }
            
        if self.filters is None:
            self.filters = {
                'time': {
                    'start': '09:30',
                    'end': '16:00',
                    'exclude_holidays': True
                },
                'volume': {
                    'min_threshold': self.min_volume,
                    'increase_factor': 1.5
                },
                'volatility': {
                    'min': 0.001,
                    'max': 0.05
                },
                'trend': {
                    'lookback': 20,
                    'min_strength': 0.6
                }
            }

class SignalGenerator:
    def __init__(self, config: Optional[SignalConfig] = None):
        """
        初始化信号生成器
        
        Args:
            config: 信号生成器配置
        """
        self.config = config or SignalConfig()
        self.validation_rules = self.load_validation_rules()
        self.signal_history = []
        
    def generate_signal(self, analysis_result: Dict) -> Optional[Dict]:
        """
        生成交易信号
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            交易信号字典或None
        """
        try:
            # 1. 验证分析结果
            validation_result = self.validate_analysis(analysis_result)
            if not validation_result['is_valid']:
                logger.info("分析结果验证未通过")
                return None
            
            # 2. 确定信号方向
            direction = self._determine_direction(analysis_result)
            if not direction:
                return None
            
            # 3. 计算关键价格水平
            entry_points = self.calculate_entry(analysis_result)
            stop_loss = self.calculate_stop_loss(analysis_result)
            take_profit = self.calculate_take_profit(analysis_result)
            
            # 4. 计算置信度
            confidence_score = self.calculate_confidence(analysis_result)
            
            # 5. 验证风险收益比
            if not self._validate_risk_reward(entry_points, stop_loss, take_profit):
                logger.info("风险收益比验证未通过")
                return None
            
            # 6. 构建信号
            signal = {
                'signal_type': 'SBS',
                'direction': direction,
                'entry_points': entry_points,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'pattern_type': analysis_result.get('pattern_type'),
                    'market_condition': analysis_result.get('market_condition'),
                    'volume_profile': analysis_result.get('volume_profile')
                }
            }
            
            # 7. 更新信号历史
            self._update_signal_history(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return None
    
    def load_validation_rules(self) -> Dict:
        """加载验证规则"""
        return {
            'basic_requirements': {
                'min_confidence': self.config.min_confidence,
                'min_volume': self.config.min_volume,
                'validation_window': self.config.validation_window
            },
            'pattern_requirements': {
                'trend_continuation': self.config.patterns['trend_continuation'],
                'trend_reversal': self.config.patterns['trend_reversal']
            },
            'filters': self.config.filters
        }
        
    def validate_analysis(self, analysis_result: Dict) -> Dict[str, Any]:
        """
        验证分析结果
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            验证结果字典，包含验证状态和详细信息
        """
        try:
            validation_result = {
                'is_valid': True,
                'details': {},
                'scores': {},
                'warnings': []
            }
            
            # 1. 基本要求验证
            basic_check = self._validate_basic_requirements(analysis_result)
            validation_result['details']['basic_requirements'] = basic_check
            if not basic_check['is_valid']:
                validation_result['is_valid'] = False
                
            # 2. 序列验证
            sequence_check = self._validate_sequence(analysis_result)
            validation_result['details']['sequence'] = sequence_check
            validation_result['scores']['sequence_quality'] = sequence_check['score']
            if not sequence_check['is_valid']:
                validation_result['is_valid'] = False
                
            # 3. 市场条件验证
            market_check = self._validate_market_conditions(analysis_result)
            validation_result['details']['market_conditions'] = market_check
            validation_result['scores']['market_quality'] = market_check['score']
            if not market_check['is_valid']:
                validation_result['is_valid'] = False
                
            # 4. 风险管理验证
            risk_check = self._validate_risk_management(analysis_result)
            validation_result['details']['risk_management'] = risk_check
            if not risk_check['is_valid']:
                validation_result['is_valid'] = False
                
            # 5. 历史模式验证
            history_check = self._validate_historical_patterns(analysis_result)
            validation_result['details']['historical_patterns'] = history_check
            validation_result['scores']['historical_quality'] = history_check['score']
            
            # 6. 实时市场状态验证
            market_state_check = self._validate_market_state(analysis_result)
            validation_result['details']['market_state'] = market_state_check
            if not market_state_check['is_valid']:
                validation_result['warnings'].append(market_state_check['warning'])
            
            # 计算总体置信度
            validation_result['confidence'] = self._calculate_overall_confidence(validation_result['scores'])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"验证分析结果失败: {e}")
            return {'is_valid': False, 'error': str(e)}
            
    def calculate_entry(self, analysis_result: Dict) -> Dict:
        """计算入场点位"""
        try:
            pattern_type = analysis_result.get('pattern_type')
            if not pattern_type:
                return None
                
            if pattern_type == 'trend_continuation':
                return self._calculate_continuation_entry(analysis_result)
            elif pattern_type == 'trend_reversal':
                return self._calculate_reversal_entry(analysis_result)
            else:
                return None
                
        except Exception as e:
            logger.error(f"计算入场点位失败: {e}")
            return None
            
    def calculate_stop_loss(self, analysis_result: Dict) -> float:
        """计算止损位"""
        try:
            pattern_type = analysis_result.get('pattern_type')
            direction = analysis_result.get('direction')
            
            if not (pattern_type and direction):
                return None
                
            # 获取关键价格点
            if pattern_type == 'trend_continuation':
                key_level = analysis_result['points'].get('point2')
            else:  # trend_reversal
                key_level = analysis_result['points'].get('point5')
                
            if not key_level:
                return None
                
            # 计算止损距离
            atr = analysis_result.get('atr', self.config.max_stop_distance)
            stop_distance = min(atr * 2, self.config.max_stop_distance)
            
            # 根据方向设置止损
            if direction == 'buy':
                return key_level - stop_distance
            else:
                return key_level + stop_distance
                
        except Exception as e:
            logger.error(f"计算止损位失败: {e}")
            return None
            
    def calculate_take_profit(self, analysis_result: Dict) -> List[float]:
        """计算获利目标"""
        try:
            entry = analysis_result.get('entry', {}).get('price')
            stop_loss = analysis_result.get('stop_loss')
            direction = analysis_result.get('direction')
            
            if not all([entry, stop_loss, direction]):
                return None
                
            # 计算风险
            risk = abs(entry - stop_loss)
            
            # 设置多个目标点
            targets = []
            reward_ratios = [2.0, 3.0, 5.0]  # 风险收益比例
            
            for ratio in reward_ratios:
                if direction == 'buy':
                    target = entry + (risk * ratio)
                else:
                    target = entry - (risk * ratio)
                targets.append(target)
                
            return targets
            
        except Exception as e:
            logger.error(f"计算获利目标失败: {e}")
            return None
            
    def calculate_confidence(self, analysis_result: Dict) -> float:
        """计算信号置信度"""
        try:
            scores = []
            
            # 1. 模式质量评分
            pattern_score = self._assess_pattern_quality(analysis_result)
            scores.append(pattern_score)
            
            # 2. 市场条件评分
            market_score = self._assess_market_condition(analysis_result)
            scores.append(market_score)
            
            # 3. 成交量特征评分
            volume_score = self._assess_volume_profile(analysis_result)
            scores.append(volume_score)
            
            # 4. 趋势一致性评分
            trend_score = self._assess_trend_alignment(analysis_result)
            scores.append(trend_score)
            
            # 5. 时间对称性评分
            time_score = self._assess_time_symmetry(analysis_result)
            scores.append(time_score)
            
            # 计算加权平均分
            weights = [0.3, 0.2, 0.2, 0.2, 0.1]
            confidence = np.average(scores, weights=weights)
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.0
            
    def _determine_direction(self, analysis_result: Dict) -> Optional[str]:
        """确定信号方向"""
        try:
            pattern_type = analysis_result.get('pattern_type')
            trend = analysis_result.get('trend', {})
            
            if not (pattern_type and trend):
                return None
                
            if pattern_type == 'trend_continuation':
                return trend.get('direction')
            elif pattern_type == 'trend_reversal':
                # 反转模式取反向
                return 'sell' if trend.get('direction') == 'buy' else 'buy'
            
            return None
            
        except Exception as e:
            logger.error(f"确定信号方向失败: {e}")
            return None
            
    def _validate_basic_requirements(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证基本要求"""
        result = {
            'is_valid': True,
            'details': {}
        }
        
        rules = self.validation_rules['basic_requirements']
        
        # 检查数据完整性
        if not all(key in analysis_result for key in ['data', 'points', 'volume', 'confidence']):
            result['is_valid'] = False
            result['details']['missing_data'] = '缺少必要的数据字段'
            return result
            
        # 检查置信度
        confidence = analysis_result.get('confidence', 0)
        if confidence < rules['min_confidence']:
            result['is_valid'] = False
            result['details']['low_confidence'] = f'置信度 {confidence} 低于最小要求 {rules["min_confidence"]}'
            
        # 检查成交量
        volume = analysis_result.get('volume', 0)
        if volume < rules['min_volume']:
            result['is_valid'] = False
            result['details']['low_volume'] = f'成交量 {volume} 低于最小要求 {rules["min_volume"]}'
            
        # 检查数据窗口
        data_length = len(analysis_result.get('data', []))
        if data_length < rules['validation_window']:
            result['is_valid'] = False
            result['details']['insufficient_data'] = f'数据点数 {data_length} 小于所需窗口 {rules["validation_window"]}'
            
        return result
        
    def _validate_sequence(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证序列"""
        result = {
            'is_valid': True,
            'score': 0.0,
            'details': {}
        }
        
        try:
            pattern_type = analysis_result.get('pattern_type')
            if not pattern_type:
                result['is_valid'] = False
                result['details']['error'] = '未指定模式类型'
                return result
                
            rules = self.validation_rules['pattern_requirements'][pattern_type]
            points = analysis_result.get('points', {})
            
            # 验证关键点完整性
            required_points = ['point1', 'point2', 'point3', 'point4', 'point5']
            missing_points = [p for p in required_points if p not in points]
            if missing_points:
                result['is_valid'] = False
                result['details']['missing_points'] = f'缺少关键点: {missing_points}'
                return result
                
            # 验证点位顺序和时间间隔
            point_scores = []
            for i in range(1, 5):
                current = points[f'point{i}']
                next_point = points[f'point{i+1}']
                
                # 时间顺序验证
                if current['timestamp'] >= next_point['timestamp']:
                    result['is_valid'] = False
                    result['details'][f'invalid_sequence_{i}'] = f'点位{i}和{i+1}时间顺序错误'
                    
                # 计算时间间隔得分
                time_diff = (next_point['timestamp'] - current['timestamp']).total_seconds()
                expected_diff = rules['expected_time_diff']
                time_score = max(0, 1 - abs(time_diff - expected_diff) / expected_diff)
                point_scores.append(time_score)
                
            # 验证价格结构
            price_structure = self._validate_price_structure(points, pattern_type)
            if not price_structure['is_valid']:
                result['is_valid'] = False
                result['details']['price_structure'] = price_structure['details']
            point_scores.append(price_structure['score'])
            
            # 验证成交量特征
            volume_profile = self._validate_volume_profile(analysis_result)
            if not volume_profile['is_valid']:
                result['is_valid'] = False
                result['details']['volume_profile'] = volume_profile['details']
            point_scores.append(volume_profile['score'])
            
            # 计算总体得分
            result['score'] = np.mean(point_scores)
            
            return result
            
        except Exception as e:
            logger.error(f"序列验证失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'details': {'error': str(e)}}
            
    def _validate_market_conditions(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证市场条件"""
        result = {
            'is_valid': True,
            'score': 0.0,
            'details': {}
        }
        
        try:
            filters = self.validation_rules['filters']
            market_data = analysis_result.get('market_data', {})
            
            # 验证时间条件
            time_check = self._validate_trading_time(market_data.get('timestamp'))
            if not time_check['is_valid']:
                result['is_valid'] = False
                result['details']['time'] = time_check['details']
                
            # 验证波动率
            volatility_check = self._validate_volatility(market_data)
            if not volatility_check['is_valid']:
                result['is_valid'] = False
                result['details']['volatility'] = volatility_check['details']
            result['score'] = volatility_check['score']
            
            # 验证流动性
            liquidity_check = self._validate_liquidity(market_data)
            if not liquidity_check['is_valid']:
                result['is_valid'] = False
                result['details']['liquidity'] = liquidity_check['details']
            result['score'] = (result['score'] + liquidity_check['score']) / 2
            
            # 验证市场趋势
            trend_check = self._validate_market_trend(market_data)
            if not trend_check['is_valid']:
                result['is_valid'] = False
                result['details']['trend'] = trend_check['details']
            result['score'] = (result['score'] + trend_check['score']) / 2
            
            return result
            
        except Exception as e:
            logger.error(f"市场条件验证失败: {e}")
            return {'is_valid': False, 'score': 0.0, 'details': {'error': str(e)}}
            
    def _validate_risk_management(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证风险管理"""
        result = {
            'is_valid': True,
            'details': {}
        }
        
        try:
            entry = analysis_result.get('entry', {}).get('price')
            stop_loss = analysis_result.get('stop_loss')
            targets = analysis_result.get('take_profit', [])
            
            if not all([entry, stop_loss, targets]):
                result['is_valid'] = False
                result['details']['missing_levels'] = '缺少必要的价格水平'
                return result
                
            # 验证止损距离
            risk = abs(entry - stop_loss)
            risk_ratio = risk / entry
            
            if risk_ratio > self.config.max_risk_ratio:
                result['is_valid'] = False
                result['details']['excessive_risk'] = f'风险比率 {risk_ratio:.2%} 超过最大限制 {self.config.max_risk_ratio:.2%}'
                
            # 验证获利目标
            for i, target in enumerate(targets, 1):
                reward = abs(target - entry)
                reward_ratio = reward / risk
                
                if reward_ratio < self.config.min_reward_ratio:
                    result['is_valid'] = False
                    result['details'][f'target_{i}_invalid'] = f'目标{i}的收益比 {reward_ratio:.2f} 低于最小要求 {self.config.min_reward_ratio}'
                    
            return result
            
        except Exception as e:
            logger.error(f"风险管理验证失败: {e}")
            return {'is_valid': False, 'details': {'error': str(e)}}
            
    def _validate_historical_patterns(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证历史模式"""
        result = {
            'is_valid': True,
            'score': 0.0,
            'details': {}
        }
        
        try:
            pattern_type = analysis_result.get('pattern_type')
            if not pattern_type or not self.signal_history:
                return result
                
            # 分析历史相似模式
            similar_patterns = []
            for signal in self.signal_history:
                if signal['metadata']['pattern_type'] == pattern_type:
                    similarity = self._calculate_pattern_similarity(analysis_result, signal)
                    if similarity > 0.8:  # 相似度阈值
                        similar_patterns.append({
                            'similarity': similarity,
                            'success': signal.get('success', False)
                        })
                        
            if similar_patterns:
                # 计算历史成功率
                success_rate = sum(1 for p in similar_patterns if p['success']) / len(similar_patterns)
                avg_similarity = np.mean([p['similarity'] for p in similar_patterns])
                
                result['score'] = (success_rate + avg_similarity) / 2
                result['details']['similar_patterns'] = len(similar_patterns)
                result['details']['success_rate'] = success_rate
                result['details']['avg_similarity'] = avg_similarity
                
            return result
            
        except Exception as e:
            logger.error(f"历史模式验证失败: {e}")
            return {'is_valid': True, 'score': 0.0, 'details': {'error': str(e)}}
            
    def _validate_market_state(self, analysis_result: Dict) -> Dict[str, Any]:
        """验证实时市场状态"""
        result = {
            'is_valid': True,
            'warning': None,
            'details': {}
        }
        
        try:
            market_state = analysis_result.get('market_state', {})
            
            # 检查重要新闻
            if market_state.get('has_important_news', False):
                result['warning'] = '重要新闻期间，建议谨慎交易'
                
            # 检查市场波动
            if market_state.get('high_volatility', False):
                result['warning'] = '市场波动剧烈，建议调整仓位'
                
            # 检查流动性
            if market_state.get('low_liquidity', False):
                result['warning'] = '市场流动性低，建议谨慎交易'
                
            # 检查价格异常
            if market_state.get('price_anomaly', False):
                result['is_valid'] = False
                result['warning'] = '检测到价格异常，建议暂停交易'
                
            return result
            
        except Exception as e:
            logger.error(f"市场状态验证失败: {e}")
            return {'is_valid': True, 'warning': None, 'details': {'error': str(e)}}
            
    def _calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """计算总体置信度"""
        try:
            weights = {
                'sequence_quality': 0.4,
                'market_quality': 0.3,
                'historical_quality': 0.3
            }
            
            weighted_scores = []
            for key, weight in weights.items():
                if key in scores:
                    weighted_scores.append(scores[key] * weight)
                    
            return sum(weighted_scores) if weighted_scores else 0.0
            
        except Exception as e:
            logger.error(f"计算总体置信度失败: {e}")
            return 0.0
        
    def _update_signal_history(self, signal: Dict):
        """更新信号历史"""
        self.signal_history.append(signal)
        if len(self.signal_history) > self.config.history_size:
            self.signal_history = self.signal_history[-self.config.history_size:]
            
    def _assess_pattern_quality(self, analysis_result: Dict) -> float:
        """评估模式质量"""
        try:
            scores = []
            
            # 1. 关键点位置评分
            points_score = analysis_result.get('points_quality', 0)
            scores.append(points_score)
            
            # 2. 形态完整性评分
            pattern_score = analysis_result.get('pattern_quality', 0)
            scores.append(pattern_score)
            
            # 3. 价格结构评分
            structure_score = analysis_result.get('price_structure', 0)
            scores.append(structure_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"模式质量评估失败: {e}")
            return 0.0
            
    def _assess_market_condition(self, analysis_result: Dict) -> float:
        """评估市场条件"""
        try:
            scores = []
            
            # 1. 趋势强度评分
            trend_score = analysis_result.get('trend_strength', 0)
            scores.append(trend_score)
            
            # 2. 波动率评分
            volatility_score = analysis_result.get('volatility_score', 0)
            scores.append(volatility_score)
            
            # 3. 市场结构评分
            structure_score = analysis_result.get('market_structure', 0)
            scores.append(structure_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"市场条件评估失败: {e}")
            return 0.0
            
    def _assess_volume_profile(self, analysis_result: Dict) -> float:
        """评估成交量特征"""
        try:
            volume_profile = analysis_result.get('volume_profile', {})
            
            # 1. 成交量趋势
            trend_aligned = volume_profile.get('trend_aligned', False)
            
            # 2. 成交量放大
            volume_increase = volume_profile.get('volume_increase', 0)
            
            # 3. 成交量分布
            distribution_score = volume_profile.get('distribution_score', 0)
            
            # 计算综合得分
            score = (float(trend_aligned) + min(volume_increase, 1.0) + distribution_score) / 3
            
            return score
            
        except Exception as e:
            logger.error(f"成交量特征评估失败: {e}")
            return 0.0
            
    def _assess_trend_alignment(self, analysis_result: Dict) -> float:
        """评估趋势一致性"""
        try:
            # 1. 获取不同时间周期的趋势
            trends = analysis_result.get('multi_timeframe_trends', {})
            if not trends:
                return 0.0
                
            # 2. 计算趋势一致性得分
            aligned_count = sum(1 for trend in trends.values() 
                              if trend['direction'] == analysis_result['direction'])
            alignment_score = aligned_count / len(trends)
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"趋势一致性评估失败: {e}")
            return 0.0
            
    def _assess_time_symmetry(self, analysis_result: Dict) -> float:
        """评估时间对称性"""
        try:
            points = analysis_result.get('points', {})
            if not points:
                return 0.0
                
            # 计算关键点之间的时间间隔
            intervals = []
            sorted_points = sorted(points.items(), key=lambda x: x[1]['time'])
            
            for i in range(len(sorted_points)-1):
                t1 = datetime.fromisoformat(sorted_points[i][1]['time'])
                t2 = datetime.fromisoformat(sorted_points[i+1][1]['time'])
                intervals.append((t2 - t1).total_seconds())
                
            if not intervals:
                return 0.0
                
            # 计算时间间隔的对称性
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            symmetry_score = 1.0 - min(std_interval / mean_interval, 1.0)
            
            return symmetry_score
            
        except Exception as e:
            logger.error(f"时间对称性评估失败: {e}")
            return 0.0 
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

from src.utils.logger import setup_logger

logger = setup_logger('data_validator')

@dataclass
class ValidationConfig:
    """验证配置"""
    price_min: float = 0.0
    price_max: float = 1e6
    volume_min: float = 0.0
    time_format: str = "%Y-%m-%d %H:%M:%S"
    required_columns: List[str] = None
    max_missing_ratio: float = 0.01
    max_duplicate_ratio: float = 0.001
    outlier_std_threshold: float = 3.0
    min_data_points: int = 100
    min_price_tick: float = 0.0001  # 最小价格变动
    max_price_change: float = 0.1  # 最大价格变动比例
    min_volume_tick: float = 1.0  # 最小成交量变动
    time_continuity_threshold: int = 5  # 时间连续性阈值(秒)
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        初始化数据验证器
        
        Args:
            config: 验证配置
        """
        self.config = config or ValidationConfig()
        self.validation_history = []
        self.logger = setup_logger('data_validator')
        
    def validate_data(self, data: Union[pd.DataFrame, Dict, np.ndarray],
                   data_type: str = 'market_data') -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            data: 输入数据
            data_type: 数据类型
            
        Returns:
            验证结果字典
        """
        try:
            # 转换数据为DataFrame
            df = self._convert_to_dataframe(data)
            
            # 初始化结果字典
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 执行验证
            self._validate_basic_requirements(df, result)
            self._validate_data_types(df, result)
            self._validate_data_ranges(df, result)
            self._validate_data_completeness(df, result)
            self._validate_data_consistency(df, result)
            self._validate_time_continuity(df, result)
            self._validate_price_movement(df, result)
            self._validate_volume_distribution(df, result)
            self._detect_outliers(df, result)
            
            # 计算统计信息
            result['statistics'] = self._calculate_statistics(df)
            
            # 更新验证历史
            self._update_validation_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'statistics': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def _convert_to_dataframe(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> pd.DataFrame:
        """转换数据为DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return pd.DataFrame(data, columns=self.config.required_columns[:data.shape[1]])
            else:
                raise ValueError("numpy数组必须是2维的")
        else:
            raise ValueError("不支持的数据类型")
    
    def _validate_basic_requirements(self, df: pd.DataFrame, result: Dict):
        """验证基本要求"""
        # 检查数据点数量
        if len(df) < self.config.min_data_points:
            result['is_valid'] = False
            result['errors'].append(
                f"数据点数量不足: {len(df)} < {self.config.min_data_points}"
            )
        
        # 检查必需列
        missing_columns = set(self.config.required_columns) - set(df.columns)
        if missing_columns:
            result['is_valid'] = False
            result['errors'].append(f"缺少必需列: {missing_columns}")
    
    def _validate_data_types(self, df: pd.DataFrame, result: Dict):
        """验证数据类型"""
        expected_types = {
            'timestamp': ['datetime64[ns]', 'object'],
            'open': ['float64', 'float32', 'int64'],
            'high': ['float64', 'float32', 'int64'],
            'low': ['float64', 'float32', 'int64'],
            'close': ['float64', 'float32', 'int64'],
            'volume': ['float64', 'float32', 'int64']
        }
        
        for col, expected in expected_types.items():
            if col in df.columns:
                if str(df[col].dtype) not in expected:
                    result['warnings'].append(
                        f"列 {col} 的数据类型不符合预期: {df[col].dtype}"
                    )
    
    def _validate_data_ranges(self, df: pd.DataFrame, result: Dict):
        """验证数据范围"""
        # 价格范围验证
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if df[col].min() < self.config.price_min:
                    result['errors'].append(f"{col}列存在小于{self.config.price_min}的值")
                    result['is_valid'] = False
                if df[col].max() > self.config.price_max:
                    result['errors'].append(f"{col}列存在大于{self.config.price_max}的值")
                    result['is_valid'] = False
        
        # 成交量范围验证
        if 'volume' in df.columns:
            if df['volume'].min() < self.config.volume_min:
                result['errors'].append(f"volume列存在小于{self.config.volume_min}的值")
                result['is_valid'] = False
    
    def _validate_data_completeness(self, df: pd.DataFrame, result: Dict):
        """验证数据完整性"""
        # 检查缺失值
        missing_ratios = df[self.config.required_columns].isnull().mean()
        for col, ratio in missing_ratios.items():
            if ratio > self.config.max_missing_ratio:
                result['errors'].append(
                    f"列 {col} 的缺失值比例过高: {ratio:.2%}"
                )
                result['is_valid'] = False
        
        # 检查重复值
        duplicate_ratio = df.duplicated().mean()
        if duplicate_ratio > self.config.max_duplicate_ratio:
            result['warnings'].append(
                f"数据重复比例过高: {duplicate_ratio:.2%}"
            )
    
    def _validate_data_consistency(self, df: pd.DataFrame, result: Dict):
        """验证数据一致性"""
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # 检查high >= low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                result['errors'].append(
                    f"发现{invalid_hl.sum()}条记录的high小于low"
                )
                result['is_valid'] = False
            
            # 检查high >= open, close
            invalid_ho = df['high'] < df['open']
            invalid_hc = df['high'] < df['close']
            if invalid_ho.any() or invalid_hc.any():
                result['errors'].append(
                    f"发现{invalid_ho.sum() + invalid_hc.sum()}条记录的high小于open或close"
                )
                result['is_valid'] = False
            
            # 检查low <= open, close
            invalid_lo = df['low'] > df['open']
            invalid_lc = df['low'] > df['close']
            if invalid_lo.any() or invalid_lc.any():
                result['errors'].append(
                    f"发现{invalid_lo.sum() + invalid_lc.sum()}条记录的low大于open或close"
                )
                result['is_valid'] = False
    
    def _validate_time_continuity(self, df: pd.DataFrame, result: Dict):
        """验证时间连续性"""
        if 'timestamp' not in df.columns:
            return
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diff = df['timestamp'].diff()
        
        # 检查时间间隔
        invalid_intervals = time_diff[time_diff.dt.total_seconds() > self.config.time_continuity_threshold]
        if not invalid_intervals.empty:
            result['warnings'].append(f"发现 {len(invalid_intervals)} 处时间不连续")
            
    def _validate_price_movement(self, df: pd.DataFrame, result: Dict):
        """验证价格变动"""
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                continue
                
            # 检查价格变动是否合理
            price_change = df[col].pct_change().abs()
            invalid_changes = price_change[price_change > self.config.max_price_change]
            
            if not invalid_changes.empty:
                result['warnings'].append(
                    f"{col} 列发现 {len(invalid_changes)} 处异常价格变动"
                )
                
    def _validate_volume_distribution(self, df: pd.DataFrame, result: Dict):
        """验证成交量分布"""
        if 'volume' not in df.columns:
            return
            
        # 检查成交量变动
        volume_changes = df['volume'].diff()
        invalid_changes = volume_changes[
            (volume_changes != 0) & 
            (volume_changes < self.config.min_volume_tick)
        ]
        
        if not invalid_changes.empty:
            result['warnings'].append(
                f"发现 {len(invalid_changes)} 处异常成交量变动"
            )
            
        # 检查成交量分布
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        outliers = df[
            (df['volume'] > volume_mean + 3 * volume_std) |
            (df['volume'] < volume_mean - 3 * volume_std)
        ]
        
        if not outliers.empty:
            result['warnings'].append(
                f"发现 {len(outliers)} 处成交量异常值"
            )
    
    def _detect_outliers(self, df: pd.DataFrame, result: Dict):
        """检测异常值"""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # 使用Z-score方法检测异常值
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.config.outlier_std_threshold
                
                if outliers.any():
                    result['warnings'].append(
                        f"列 {col} 发现 {outliers.sum()} 个异常值"
                    )
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """计算统计信息"""
        stats = {
            'record_count': len(df),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'column_stats': {}
        }
        
        for col in self.config.required_columns:
            if col in df.columns and col != 'timestamp':
                stats['column_stats'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'missing_ratio': float(df[col].isnull().mean())
                }
        
        return stats
    
    def _update_validation_history(self, result: Dict):
        """更新验证历史"""
        self.validation_history.append(result)
        
        # 保持历史记录在合理范围内
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def save_validation_result(self, result: Dict, filename: Optional[str] = None):
        """保存验证结果"""
        if filename is None:
            filename = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        try:
            os.makedirs('validation_results', exist_ok=True)
            filepath = os.path.join('validation_results', filename)
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"验证结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
    
    def get_validation_summary(self) -> Dict:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {}
            
        summary = {
            'total_validations': len(self.validation_history),
            'success_rate': sum(1 for r in self.validation_history if r['is_valid']) / len(self.validation_history),
            'common_errors': {},
            'common_warnings': {}
        }
        
        # 统计常见错误和警告
        for result in self.validation_history:
            for error in result['errors']:
                summary['common_errors'][error] = summary['common_errors'].get(error, 0) + 1
            for warning in result['warnings']:
                summary['common_warnings'][warning] = summary['common_warnings'].get(warning, 0) + 1
        
        # 排序并只保留前10个最常见的问题
        summary['common_errors'] = dict(
            sorted(summary['common_errors'].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        summary['common_warnings'] = dict(
            sorted(summary['common_warnings'].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary 
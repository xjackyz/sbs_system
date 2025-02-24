"""
性能追踪器模块，用于监控和记录系统性能指标
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger('performance_tracker')

class PerformanceTracker:
    """性能追踪器类"""
    
    def __init__(self, save_dir: str = 'results/performance'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory_usage': [],
            'disk_usage': [],
            'latency': [],
            'throughput': [],
            'error_rate': [],
            'success_rate': []
        }
        
        self.timestamps = []
        
    def record_metric(self, metric_name: str, value: float):
        """记录单个指标"""
        if metric_name not in self.metrics:
            logger.warning(f"未知指标: {metric_name}")
            return
            
        self.metrics[metric_name].append(value)
        if len(self.timestamps) < len(self.metrics[metric_name]):
            self.timestamps.append(datetime.now())
            
    def record_metrics(self, metrics: Dict[str, float]):
        """记录多个指标"""
        for metric_name, value in metrics.items():
            self.record_metric(metric_name, value)
            
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        if metric_name not in self.metrics:
            return {}
            
        values = np.array(self.metrics[metric_name])
        if len(values) == 0:
            return {}
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'last': float(values[-1])
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的统计信息"""
        return {
            metric_name: self.get_metric_stats(metric_name)
            for metric_name in self.metrics
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        df = pd.DataFrame(self.metrics)
        df.index = pd.to_datetime(self.timestamps)
        return df
        
    def save(self, filename: Optional[str] = None):
        """保存性能数据"""
        if filename is None:
            filename = f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        save_path = self.save_dir / filename
        
        try:
            data = {
                'metrics': self.metrics,
                'timestamps': [ts.isoformat() for ts in self.timestamps]
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"性能数据已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"保存性能数据失败: {e}")
            
    def load(self, filename: str):
        """加载性能数据"""
        load_path = self.save_dir / filename
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
                
            self.metrics = data['metrics']
            self.timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
            
            logger.info(f"性能数据已从{load_path}加载")
            
        except Exception as e:
            logger.error(f"加载性能数据失败: {e}")
            
    def clear(self):
        """清除所有记录的指标"""
        for metric_name in self.metrics:
            self.metrics[metric_name] = []
        self.timestamps = []
        logger.info("所有性能指标已清除") 
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from queue import Queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ProcessorConfig:
    """数据处理器配置"""
    batch_size: int = 1000
    max_queue_size: int = 5000
    num_workers: int = 4
    cache_size: int = 1000
    cleanup_interval: int = 3600  # 清理间隔（秒）

class DataProcessor:
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.logger = logging.getLogger('data_processor')
        self.data_queue = Queue(maxsize=self.config.max_queue_size)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.is_running = False
        self.processor_thread = None
        self.cleanup_thread = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
    def start(self):
        """启动处理器"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # 启动处理线程
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        self.logger.info("数据处理器已启动")
        
    def stop(self):
        """停止处理器"""
        self.is_running = False
        if self.processor_thread:
            self.processor_thread.join()
        if self.cleanup_thread:
            self.cleanup_thread.join()
        self.executor.shutdown()
        self.logger.info("数据处理器已停止")
        
    def add_data(self, data: pd.DataFrame, key: str = None):
        """添加数据到处理队列"""
        try:
            if key and self._check_cache(key):
                self.logger.debug(f"数据已存在于缓存中: {key}")
                return
                
            self.data_queue.put((data, key))
            self.logger.debug(f"数据已添加到处理队列: {key}")
            
        except Exception as e:
            self.logger.error(f"添加数据失败: {e}")
            
    def _processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 批量获取数据
                batch = []
                while len(batch) < self.config.batch_size and not self.data_queue.empty():
                    batch.append(self.data_queue.get_nowait())
                    
                if not batch:
                    time.sleep(0.1)
                    continue
                    
                # 并行处理数据
                futures = []
                for data, key in batch:
                    future = self.executor.submit(self._process_data, data, key)
                    futures.append(future)
                    
                # 等待所有处理完成
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            self._update_cache(result)
                    except Exception as e:
                        self.logger.error(f"数据处理失败: {e}")
                        
            except Exception as e:
                self.logger.error(f"处理循环出错: {e}")
                
    def _process_data(self, data: pd.DataFrame, key: str = None) -> Optional[Tuple[str, Dict]]:
        """处理单个数据项"""
        try:
            # 1. 数据清洗
            cleaned_data = self._clean_data(data)
            
            # 2. 特征计算
            features = self._calculate_features(cleaned_data)
            
            # 3. 数据标准化
            normalized_data = self._normalize_data(features)
            
            # 4. 生成处理结果
            result = {
                'processed_data': normalized_data,
                'timestamp': pd.Timestamp.now().isoformat(),
                'features': features
            }
            
            return (key, result) if key else None
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            return None
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        try:
            # 1. 删除重复行
            data = data.drop_duplicates()
            
            # 2. 处理缺失值
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # 3. 删除异常值
            data = self._remove_outliers(data)
            
            # 4. 确保时间索引
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            return data
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {e}")
            return data
            
    def _calculate_features(self, data: pd.DataFrame) -> Dict:
        """计算特征"""
        try:
            features = {}
            
            # 基本统计特征
            features['basic_stats'] = {
                'mean': data.mean().to_dict(),
                'std': data.std().to_dict(),
                'min': data.min().to_dict(),
                'max': data.max().to_dict()
            }
            
            # 技术指标
            if 'close' in data.columns:
                features['technical'] = {
                    'sma_20': data['close'].rolling(20).mean().iloc[-1],
                    'sma_50': data['close'].rolling(50).mean().iloc[-1],
                    'volatility': data['close'].pct_change().std(),
                }
                
            return features
            
        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
            return {}
            
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        try:
            # 使用Z-score标准化
            normalized = (data - data.mean()) / data.std()
            return normalized
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {e}")
            return data
            
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """删除异常值"""
        try:
            # 使用IQR方法检测异常值
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值范围
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 移除异常值
            mask = ~((data < lower_bound) | (data > upper_bound)).any(axis=1)
            return data[mask]
            
        except Exception as e:
            self.logger.error(f"异常值删除失败: {e}")
            return data
            
    def _check_cache(self, key: str) -> bool:
        """检查缓存"""
        with self.cache_lock:
            return key in self.cache
            
    def _update_cache(self, result: Tuple[str, Dict]):
        """更新缓存"""
        try:
            key, data = result
            with self.cache_lock:
                self.cache[key] = data
                
                # 如果缓存超出大小限制，删除最旧的数据
                if len(self.cache) > self.config.cache_size:
                    oldest_key = min(self.cache.keys(), 
                                   key=lambda k: self.cache[k]['timestamp'])
                    del self.cache[oldest_key]
                    
        except Exception as e:
            self.logger.error(f"缓存更新失败: {e}")
            
    def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_cache()
                
            except Exception as e:
                self.logger.error(f"清理过程出错: {e}")
                
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            with self.cache_lock:
                current_time = pd.Timestamp.now()
                keys_to_remove = []
                
                for key, data in self.cache.items():
                    # 删除超过24小时的数据
                    data_time = pd.Timestamp(data['timestamp'])
                    if (current_time - data_time).total_seconds() > 86400:
                        keys_to_remove.append(key)
                        
                for key in keys_to_remove:
                    del self.cache[key]
                    
                self.logger.info(f"已清理 {len(keys_to_remove)} 条缓存数据")
                
        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")
            
    def get_processed_data(self, key: str) -> Optional[Dict]:
        """获取处理后的数据"""
        try:
            with self.cache_lock:
                return self.cache.get(key)
                
        except Exception as e:
            self.logger.error(f"获取处理数据失败: {e}")
            return None 
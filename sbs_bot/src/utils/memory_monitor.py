import os
import gc
import psutil
import torch
import logging
import threading
import time
import numpy as np
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """内存使用监控工具"""
    
    def __init__(self, interval: float = 1.0):
        """
        初始化内存监控器
        
        Args:
            interval: 监控间隔时间（秒）
        """
        self.interval = interval
        self.memory_usage = []
        self._monitor_thread = None
        self._stop_flag = threading.Event()
        
    def _monitor(self):
        """监控线程的主循环"""
        while not self._stop_flag.is_set():
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # 转换为MB
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_usage.append(memory_mb)
            
            time.sleep(self.interval)
            
    def start(self):
        """开始监控"""
        if self._monitor_thread is None:
            self._stop_flag.clear()
            self._monitor_thread = threading.Thread(target=self._monitor)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
    def stop(self):
        """停止监控"""
        if self._monitor_thread is not None:
            self._stop_flag.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            
    def get_stats(self) -> Dict[str, float]:
        """
        获取内存使用统计信息
        
        Returns:
            包含统计信息的字典：
            - avg_memory_usage: 平均内存使用（MB）
            - peak_memory_usage: 峰值内存使用（MB）
            - min_memory_usage: 最小内存使用（MB）
            - std_memory_usage: 内存使用标准差（MB）
        """
        if not self.memory_usage:
            return {
                'avg_memory_usage': 0.0,
                'peak_memory_usage': 0.0,
                'min_memory_usage': 0.0,
                'std_memory_usage': 0.0
            }
            
        return {
            'avg_memory_usage': np.mean(self.memory_usage),
            'peak_memory_usage': np.max(self.memory_usage),
            'min_memory_usage': np.min(self.memory_usage),
            'std_memory_usage': np.std(self.memory_usage)
        }
        
    def reset(self):
        """重置监控数据"""
        self.memory_usage = []
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {}
            
        try:
            torch.cuda.synchronize()
            
            # 获取当前设备的内存信息
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            # 计算使用率
            memory_used = memory_allocated / memory_total
            memory_reserved_ratio = memory_reserved / memory_total
            
            return {
                'allocated_mb': memory_allocated / 1024 / 1024,
                'reserved_mb': memory_reserved / 1024 / 1024,
                'total_mb': memory_total / 1024 / 1024,
                'used_ratio': memory_used,
                'reserved_ratio': memory_reserved_ratio
            }
            
        except Exception as e:
            logger.error(f"获取GPU内存信息失败: {e}")
            return {}
            
    def get_system_memory_info(self) -> Dict[str, float]:
        """获取系统内存信息"""
        try:
            memory = psutil.virtual_memory()
            
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'used_ratio': memory.percent / 100
            }
            
        except Exception as e:
            logger.error(f"获取系统内存信息失败: {e}")
            return {}
            
    def check_memory(self, force_gc: bool = True) -> Dict[str, Dict[str, float]]:
        """
        检查内存使用情况
        
        Args:
            force_gc: 是否强制执行垃圾回收
            
        Returns:
            包含GPU和系统内存信息的字典
        """
        # 获取内存信息
        gpu_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()
        
        # 记录时间戳
        timestamp = datetime.now().isoformat()
        
        # 合并信息
        memory_info = {
            'timestamp': timestamp,
            'gpu': gpu_info,
            'system': system_info
        }
        
        # 添加到历史记录
        self.memory_usage.append(system_info['used_mb'])
        
        # 检查是否超过阈值
        if gpu_info and gpu_info.get('used_ratio', 0) > 0.8:
            logger.warning(f"GPU内存使用率过高: {gpu_info['used_ratio']*100:.1f}%")
            if force_gc:
                self.clean_memory()
                
        if system_info.get('used_ratio', 0) > 0.8:
            logger.warning(f"系统内存使用率过高: {system_info['used_ratio']*100:.1f}%")
            if force_gc:
                self.clean_memory()
                
        return {
            'gpu': gpu_info,
            'system': system_info
        }
        
    def clean_memory(self):
        """清理内存"""
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 强制垃圾回收
        gc.collect()
        
    def save_stats(self, filename: Optional[str] = None):
        """
        保存内存统计信息到文件
        
        Args:
            filename: 保存的文件名,如果为None则使用时间戳
        """
        if not filename:
            filename = f"memory_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
        filepath = os.path.join("logs", filename)
        
        try:
            with open(filepath, 'w') as f:
                for stat in self.memory_usage:
                    f.write(f"{stat}\n")
            logger.info(f"内存统计信息已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存内存统计信息失败: {e}")
            
    def get_peak_memory(self) -> Dict[str, float]:
        """获取峰值内存使用情况"""
        if not self.memory_usage:
            return {}
            
        peak_memory = np.max(self.memory_usage)
        
        return {
            'peak_memory_mb': peak_memory
        }
        
    def reset_stats(self):
        """重置统计信息"""
        self.memory_usage = [] 
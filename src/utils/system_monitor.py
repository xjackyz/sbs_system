import psutil
import os
import time
import threading
import json
from typing import Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass
import torch
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger('system_monitor')

@dataclass
class MonitorConfig:
    """监控配置"""
    cpu_threshold: float = 90.0  # CPU使用率阈值
    memory_threshold: float = 85.0  # 内存使用率阈值
    gpu_memory_threshold: float = 85.0  # GPU显存使用率阈值
    disk_threshold: float = 90.0  # 磁盘使用率阈值
    check_interval: int = 60  # 检查间隔（秒）
    history_size: int = 1000  # 历史记录大小
    alert_cooldown: int = 300  # 警报冷却时间（秒）
    log_dir: str = "logs/monitor"  # 监控日志目录

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        初始化系统监控器
        
        Args:
            config: 监控配置
        """
        self.config = config or MonitorConfig()
        self.history = []
        self.alert_callbacks = []
        self._monitor_thread = None
        self._stop_flag = threading.Event()
        self.last_alert_time = {}
        
        # 创建日志目录
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def start(self):
        """启动监控"""
        if self._monitor_thread is None:
            self._stop_flag.clear()
            self._monitor_thread = threading.Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("系统监控已启动")
            
    def stop(self):
        """停止监控"""
        if self._monitor_thread is not None:
            self._stop_flag.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            logger.info("系统监控已停止")
            
    def add_alert_callback(self, callback: Callable):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_flag.is_set():
            try:
                # 检查系统状态
                status = self.check_system_status()
                
                # 更新历史记录
                self._update_history(status)
                
                # 检查是否需要发出警报
                self._check_alerts(status)
                
                # 保存监控数据
                self._save_monitoring_data(status)
                
                # 等待下次检查
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"监控过程出错: {e}")
                time.sleep(5)  # 出错后短暂等待
                
    def check_system_status(self) -> Dict:
        """
        检查系统状态
        
        Returns:
            系统状态信息
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._check_cpu_status(),
            'memory': self._check_memory_status(),
            'disk': self._check_disk_status(),
            'network': self._check_network_status(),
            'process': self._check_process_status()
        }
        
        # 如果有GPU，检查GPU状态
        if torch.cuda.is_available():
            status['gpu'] = self._check_gpu_status()
            
        return status
    
    def _check_cpu_status(self) -> Dict:
        """检查CPU状态"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            'usage_per_cpu': cpu_percent,
            'average_usage': sum(cpu_percent) / len(cpu_percent),
            'frequency': {
                'current': cpu_freq.current,
                'min': cpu_freq.min,
                'max': cpu_freq.max
            }
        }
    
    def _check_memory_status(self) -> Dict:
        """检查内存状态"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),
            'used': memory.used / (1024**3),
            'percent': memory.percent,
            'swap': {
                'total': swap.total / (1024**3),
                'used': swap.used / (1024**3),
                'percent': swap.percent
            }
        }
    
    def _check_disk_status(self) -> Dict:
        """检查磁盘状态"""
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total': usage.total / (1024**3),
                    'used': usage.used / (1024**3),
                    'free': usage.free / (1024**3),
                    'percent': usage.percent
                }
            except Exception:
                continue
                
        return disk_usage
    
    def _check_network_status(self) -> Dict:
        """检查网络状态"""
        net_io = psutil.net_io_counters()
        
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'error_in': net_io.errin,
            'error_out': net_io.errout,
            'drop_in': net_io.dropin,
            'drop_out': net_io.dropout
        }
    
    def _check_process_status(self) -> Dict:
        """检查进程状态"""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads(),
            'fds': process.num_fds(),
            'status': process.status()
        }
    
    def _check_gpu_status(self) -> Dict:
        """检查GPU状态"""
        gpu_stats = []
        
        for i in range(torch.cuda.device_count()):
            gpu_stats.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3),
                'memory_reserved': torch.cuda.memory_reserved(i) / (1024**3),
                'max_memory_allocated': torch.cuda.max_memory_allocated(i) / (1024**3),
                'utilization': torch.cuda.utilization(i)
            })
            
        return gpu_stats
    
    def _update_history(self, status: Dict):
        """更新历史记录"""
        self.history.append(status)
        
        # 保持历史记录大小在限制范围内
        while len(self.history) > self.config.history_size:
            self.history.pop(0)
            
    def _check_alerts(self, status: Dict):
        """检查是否需要发出警报"""
        current_time = time.time()
        alerts = []
        
        # 检查CPU使用率
        if status['cpu']['average_usage'] > self.config.cpu_threshold:
            alerts.append({
                'type': 'CPU_HIGH',
                'message': f"CPU使用率过高: {status['cpu']['average_usage']:.1f}%"
            })
            
        # 检查内存使用率
        if status['memory']['percent'] > self.config.memory_threshold:
            alerts.append({
                'type': 'MEMORY_HIGH',
                'message': f"内存使用率过高: {status['memory']['percent']:.1f}%"
            })
            
        # 检查磁盘使用率
        for mount_point, disk_info in status['disk'].items():
            if disk_info['percent'] > self.config.disk_threshold:
                alerts.append({
                    'type': 'DISK_HIGH',
                    'message': f"磁盘使用率过高 ({mount_point}): {disk_info['percent']:.1f}%"
                })
                
        # 检查GPU使用率
        if 'gpu' in status:
            for gpu in status['gpu']:
                memory_used = gpu['memory_allocated'] / gpu['memory_reserved'] * 100
                if memory_used > self.config.gpu_memory_threshold:
                    alerts.append({
                        'type': 'GPU_MEMORY_HIGH',
                        'message': f"GPU {gpu['id']} 显存使用率过高: {memory_used:.1f}%"
                    })
                    
        # 触发警报
        for alert in alerts:
            self._trigger_alert(alert, current_time)
            
    def _trigger_alert(self, alert: Dict, current_time: float):
        """触发警报"""
        alert_type = alert['type']
        
        # 检查是否在冷却期
        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < self.config.alert_cooldown:
                return
                
        # 更新最后警报时间
        self.last_alert_time[alert_type] = current_time
        
        # 记录警报
        logger.warning(f"系统警报: {alert['message']}")
        
        # 调用警报回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警报回调执行失败: {e}")
                
    def _save_monitoring_data(self, status: Dict):
        """保存监控数据"""
        try:
            filename = os.path.join(
                self.config.log_dir,
                f"monitor_{datetime.now().strftime('%Y%m%d')}.json"
            )
            
            with open(filename, 'a') as f:
                f.write(json.dumps(status) + '\n')
                
        except Exception as e:
            logger.error(f"保存监控数据失败: {e}")
            
    def get_system_metrics(self) -> Dict:
        """
        获取系统指标统计信息
        
        Returns:
            系统指标统计
        """
        if not self.history:
            return {}
            
        metrics = {
            'cpu': {
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0
            },
            'memory': {
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0
            }
        }
        
        # 计算统计值
        for status in self.history:
            # CPU统计
            cpu_usage = status['cpu']['average_usage']
            metrics['cpu']['min'] = min(metrics['cpu']['min'], cpu_usage)
            metrics['cpu']['max'] = max(metrics['cpu']['max'], cpu_usage)
            metrics['cpu']['avg'] += cpu_usage
            
            # 内存统计
            memory_usage = status['memory']['percent']
            metrics['memory']['min'] = min(metrics['memory']['min'], memory_usage)
            metrics['memory']['max'] = max(metrics['memory']['max'], memory_usage)
            metrics['memory']['avg'] += memory_usage
            
        # 计算平均值
        history_len = len(self.history)
        metrics['cpu']['avg'] /= history_len
        metrics['memory']['avg'] /= history_len
        
        return metrics 
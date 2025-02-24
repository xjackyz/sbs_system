"""
系统监控模块
"""
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable
import psutil
import os
import time
import json
from datetime import datetime
import threading
from collections import deque

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
        self.config = config or MonitorConfig()
        self.history = deque(maxlen=self.config.history_size)
        self.alert_callbacks = []
        self.last_alert_time = {}
        self.running = False
        self.monitor_thread = None
        self._setup_log_dir()
        
    def _setup_log_dir(self):
        """创建日志目录"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def start(self):
        """启动监控"""
        if self.running:
            logger.warning("监控器已经在运行")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("系统监控器已启动")
        
    def stop(self):
        """停止监控"""
        if not self.running:
            return
            
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("系统监控器已停止")
        
    def add_alert_callback(self, callback: Callable):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                status = self.check_system_status()
                self._update_history(status)
                self._check_alerts(status)
                self._save_monitoring_data(status)
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5)  # 错误后短暂暂停
                
    def check_system_status(self) -> Dict:
        """检查系统状态"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'cpu': self._check_cpu_status(),
                'memory': self._check_memory_status(),
                'disk': self._check_disk_status(),
                'network': self._check_network_status(),
                'process': self._check_process_status()
            }
            
            # 如果有GPU，添加GPU状态
            gpu_status = self._check_gpu_status()
            if gpu_status:
                status['gpu'] = gpu_status
                
            return status
            
        except Exception as e:
            logger.error(f"检查系统状态失败: {e}")
            return {}
            
    def _check_cpu_status(self) -> Dict:
        """检查CPU状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        return {
            'usage': cpu_percent,
            'count': cpu_count,
            'frequency': cpu_freq.current if cpu_freq else None
        }
        
    def _check_memory_status(self) -> Dict:
        """检查内存状态"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
        
    def _check_disk_status(self) -> Dict:
        """检查磁盘状态"""
        disk = psutil.disk_usage('/')
        io_counters = psutil.disk_io_counters()
        
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent,
            'read_bytes': io_counters.read_bytes if io_counters else None,
            'write_bytes': io_counters.write_bytes if io_counters else None
        }
        
    def _check_network_status(self) -> Dict:
        """检查网络状态"""
        net = psutil.net_io_counters()
        
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
            'errin': net.errin,
            'errout': net.errout
        }
        
    def _check_process_status(self) -> Dict:
        """检查进程状态"""
        current_process = psutil.Process()
        
        return {
            'pid': current_process.pid,
            'cpu_percent': current_process.cpu_percent(),
            'memory_percent': current_process.memory_percent(),
            'num_threads': current_process.num_threads(),
            'status': current_process.status()
        }
        
    def _check_gpu_status(self) -> Optional[Dict]:
        """检查GPU状态"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved()
            }
            
        except ImportError:
            return None
            
    def _update_history(self, status: Dict):
        """更新历史记录"""
        self.history.append(status)
        
    def _check_alerts(self, status: Dict):
        """检查是否需要发出警报"""
        current_time = time.time()
        alerts = []
        
        # 检查CPU使用率
        if status.get('cpu', {}).get('usage', 0) > self.config.cpu_threshold:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"CPU使用率过高: {status['cpu']['usage']}%"
            })
            
        # 检查内存使用率
        if status.get('memory', {}).get('percent', 0) > self.config.memory_threshold:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"内存使用率过高: {status['memory']['percent']}%"
            })
            
        # 检查磁盘使用率
        if status.get('disk', {}).get('percent', 0) > self.config.disk_threshold:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"磁盘使用率过高: {status['disk']['percent']}%"
            })
            
        # 检查GPU使用率
        if status.get('gpu'):
            memory_used = status['gpu'].get('memory_allocated', 0)
            memory_total = status['gpu'].get('memory_reserved', 0)
            if memory_total > 0:
                gpu_usage = (memory_used / memory_total) * 100
                if gpu_usage > self.config.gpu_memory_threshold:
                    alerts.append({
                        'type': 'gpu',
                        'level': 'warning',
                        'message': f"GPU显存使用率过高: {gpu_usage:.1f}%"
                    })
                    
        # 触发警报
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['level']}"
            last_alert = self.last_alert_time.get(alert_key, 0)
            
            if current_time - last_alert >= self.config.alert_cooldown:
                self._trigger_alert(alert, current_time)
                
    def _trigger_alert(self, alert: Dict, current_time: float):
        """触发警报"""
        alert_key = f"{alert['type']}_{alert['level']}"
        self.last_alert_time[alert_key] = current_time
        
        # 记录警报
        logger.warning(alert['message'])
        
        # 调用警报回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警报回调函数执行失败: {e}")
                
    def _save_monitoring_data(self, status: Dict):
        """保存监控数据"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"monitor_{timestamp}.json"
            filepath = os.path.join(self.config.log_dir, filename)
            
            # 保存数据
            with open(filepath, 'a') as f:
                json.dump(status, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"保存监控数据失败: {e}")
            
    def get_system_metrics(self) -> Dict:
        """获取系统指标统计"""
        if not self.history:
            return {}
            
        try:
            metrics = {
                'cpu_usage_avg': sum(s['cpu']['usage'] for s in self.history) / len(self.history),
                'memory_usage_avg': sum(s['memory']['percent'] for s in self.history) / len(self.history),
                'disk_usage_avg': sum(s['disk']['percent'] for s in self.history) / len(self.history)
            }
            
            if any('gpu' in s for s in self.history):
                gpu_metrics = [s['gpu'] for s in self.history if 'gpu' in s]
                metrics['gpu_memory_used_avg'] = sum(g['memory_allocated'] for g in gpu_metrics) / len(gpu_metrics)
                
            return metrics
            
        except Exception as e:
            logger.error(f"计算系统指标失败: {e}")
            return {} 
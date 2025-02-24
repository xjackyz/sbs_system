"""
系统监控模块的单元测试
"""
import unittest
from unittest.mock import patch, MagicMock
import time
from src.monitor.system_monitor import SystemMonitor, MonitorConfig

class TestSystemMonitor(unittest.TestCase):
    """测试系统监控器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = MonitorConfig(
            cpu_threshold=80.0,
            memory_threshold=75.0,
            gpu_memory_threshold=75.0,
            disk_threshold=80.0,
            check_interval=1,
            history_size=10,
            alert_cooldown=5,
            log_dir="tests/logs/monitor"
        )
        self.monitor = SystemMonitor(self.config)
        
    def tearDown(self):
        """测试后清理"""
        if self.monitor.running:
            self.monitor.stop()
            
    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    def test_check_cpu_status(self, mock_cpu_freq, mock_cpu_count, mock_cpu_percent):
        """测试CPU状态检查"""
        # 设置模拟返回值
        mock_cpu_percent.return_value = 50.0
        mock_cpu_count.return_value = 8
        mock_cpu_freq.return_value = MagicMock(current=2400.0)
        
        # 执行检查
        status = self.monitor._check_cpu_status()
        
        # 验证结果
        self.assertEqual(status['usage'], 50.0)
        self.assertEqual(status['count'], 8)
        self.assertEqual(status['frequency'], 2400.0)
        
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_check_memory_status(self, mock_swap_memory, mock_virtual_memory):
        """测试内存状态检查"""
        # 设置模拟返回值
        mock_virtual_memory.return_value = MagicMock(
            total=16000000000,
            available=8000000000,
            percent=50.0
        )
        mock_swap_memory.return_value = MagicMock(
            total=8000000000,
            used=1000000000,
            percent=12.5
        )
        
        # 执行检查
        status = self.monitor._check_memory_status()
        
        # 验证结果
        self.assertEqual(status['total'], 16000000000)
        self.assertEqual(status['available'], 8000000000)
        self.assertEqual(status['percent'], 50.0)
        self.assertEqual(status['swap_total'], 8000000000)
        self.assertEqual(status['swap_used'], 1000000000)
        self.assertEqual(status['swap_percent'], 12.5)
        
    def test_alert_system(self):
        """测试警报系统"""
        # 创建模拟回调函数
        alerts = []
        def alert_callback(alert):
            alerts.append(alert)
            
        # 添加回调函数
        self.monitor.add_alert_callback(alert_callback)
        
        # 模拟高CPU使用率
        with patch('src.monitor.system_monitor.SystemMonitor._check_cpu_status') as mock_cpu:
            mock_cpu.return_value = {'usage': 90.0}
            
            # 触发警报检查
            status = {'cpu': {'usage': 90.0}}
            self.monitor._check_alerts(status)
            
            # 验证警报
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]['type'], 'cpu')
            self.assertEqual(alerts[0]['level'], 'warning')
            
    def test_history_management(self):
        """测试历史记录管理"""
        # 添加测试数据
        test_data = [
            {'cpu': {'usage': i}, 'memory': {'percent': i}} 
            for i in range(15)
        ]
        
        # 验证历史记录大小限制
        for data in test_data:
            self.monitor._update_history(data)
            
        self.assertEqual(len(self.monitor.history), 10)  # history_size=10
        self.assertEqual(self.monitor.history[-1]['cpu']['usage'], 14)
        
    def test_system_metrics(self):
        """测试系统指标统计"""
        # 添加测试数据
        test_data = [
            {
                'cpu': {'usage': 50.0},
                'memory': {'percent': 60.0},
                'disk': {'percent': 70.0}
            },
            {
                'cpu': {'usage': 60.0},
                'memory': {'percent': 70.0},
                'disk': {'percent': 80.0}
            }
        ]
        
        for data in test_data:
            self.monitor._update_history(data)
            
        # 获取指标统计
        metrics = self.monitor.get_system_metrics()
        
        # 验证结果
        self.assertEqual(metrics['cpu_usage_avg'], 55.0)
        self.assertEqual(metrics['memory_usage_avg'], 65.0)
        self.assertEqual(metrics['disk_usage_avg'], 75.0)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_gpu_status(
        self,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_current_device,
        mock_device_count,
        mock_is_available
    ):
        """测试GPU状态检查"""
        # 设置模拟返回值
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        mock_current_device.return_value = 0
        mock_memory_allocated.return_value = 4000000000
        mock_memory_reserved.return_value = 8000000000
        
        # 执行检查
        status = self.monitor._check_gpu_status()
        
        # 验证结果
        self.assertIsNotNone(status)
        self.assertEqual(status['device_count'], 2)
        self.assertEqual(status['current_device'], 0)
        self.assertEqual(status['memory_allocated'], 4000000000)
        self.assertEqual(status['memory_reserved'], 8000000000)
        
if __name__ == '__main__':
    unittest.main() 
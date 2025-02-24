import unittest
import psutil
import json
from unittest.mock import Mock, patch
from datetime import datetime
from src.monitor.system_monitor import SystemMonitor, MonitorConfig
from src.monitor.discord_notifier import DiscordNotifier
from src.monitor.performance_tracker import PerformanceTracker

class TestMonitor(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = MonitorConfig(
            cpu_threshold=80,
            memory_threshold=85,
            disk_threshold=90,
            gpu_threshold=75,
            check_interval=60,
            log_interval=300,
            alert_cooldown=1800,
            discord_webhook_url="https://discord.com/api/webhooks/test",
            enable_performance_tracking=True,
            enable_error_tracking=True,
            enable_resource_tracking=True
        )
        
        self.monitor = SystemMonitor(self.config)
        self.notifier = DiscordNotifier(self.config.discord_webhook_url)
        self.tracker = PerformanceTracker()
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.monitor.config.cpu_threshold, 80)
        self.assertEqual(self.monitor.config.memory_threshold, 85)
        self.assertTrue(self.monitor.config.enable_performance_tracking)
        
    @patch('psutil.cpu_percent')
    def test_check_cpu(self, mock_cpu):
        """测试CPU检查"""
        mock_cpu.return_value = 90
        status = self.monitor.check_cpu()
        
        self.assertIsInstance(status, dict)
        self.assertIn('usage', status)
        self.assertIn('alert', status)
        self.assertTrue(status['alert'])
        
    @patch('psutil.virtual_memory')
    def test_check_memory(self, mock_memory):
        """测试内存检查"""
        mock_memory.return_value = Mock(
            percent=95,
            total=16*1024*1024*1024,
            available=2*1024*1024*1024
        )
        
        status = self.monitor.check_memory()
        
        self.assertIsInstance(status, dict)
        self.assertIn('usage', status)
        self.assertIn('alert', status)
        self.assertTrue(status['alert'])
        
    @patch('psutil.disk_usage')
    def test_check_disk(self, mock_disk):
        """测试磁盘检查"""
        mock_disk.return_value = Mock(
            percent=95,
            total=500*1024*1024*1024,
            used=475*1024*1024*1024
        )
        
        status = self.monitor.check_disk('/')
        
        self.assertIsInstance(status, dict)
        self.assertIn('usage', status)
        self.assertIn('alert', status)
        self.assertTrue(status['alert'])
        
    @patch('src.monitor.system_monitor.GPUtil')
    def test_check_gpu(self, mock_gpu):
        """测试GPU检查"""
        mock_gpu.getGPUs.return_value = [
            Mock(load=0.8, memoryUtil=0.7, temperature=75)
        ]
        
        status = self.monitor.check_gpu()
        
        self.assertIsInstance(status, dict)
        self.assertIn('usage', status)
        self.assertIn('memory', status)
        self.assertIn('temperature', status)
        self.assertTrue(status['alert'])
        
    def test_check_process(self):
        """测试进程检查"""
        process = psutil.Process()
        status = self.monitor.check_process(process)
        
        self.assertIsInstance(status, dict)
        self.assertIn('cpu_percent', status)
        self.assertIn('memory_percent', status)
        self.assertIn('status', status)
        
    @patch('requests.post')
    def test_send_alert(self, mock_post):
        """测试告警发送"""
        mock_post.return_value = Mock(status_code=200)
        
        alert_data = {
            'type': 'cpu_alert',
            'message': 'CPU usage exceeds threshold',
            'value': 90,
            'threshold': 80,
            'timestamp': datetime.now().isoformat()
        }
        
        response = self.notifier.send_alert(alert_data)
        self.assertTrue(response)
        
    def test_track_performance(self):
        """测试性能追踪"""
        metrics = {
            'response_time': 100,
            'throughput': 1000,
            'error_rate': 0.01
        }
        
        self.tracker.track_metrics(metrics)
        
        tracked_data = self.tracker.get_metrics()
        self.assertIn('response_time', tracked_data)
        self.assertIn('throughput', tracked_data)
        self.assertIn('error_rate', tracked_data)
        
    def test_calculate_statistics(self):
        """测试统计计算"""
        metrics = [
            {'response_time': 100, 'throughput': 1000},
            {'response_time': 150, 'throughput': 900},
            {'response_time': 120, 'throughput': 950}
        ]
        
        stats = self.tracker.calculate_statistics(metrics)
        
        self.assertIn('response_time_avg', stats)
        self.assertIn('throughput_avg', stats)
        self.assertIn('response_time_max', stats)
        self.assertIn('throughput_min', stats)
        
    def test_generate_report(self):
        """测试报告生成"""
        self.tracker.track_metrics({
            'response_time': 100,
            'throughput': 1000,
            'error_rate': 0.01
        })
        
        report = self.tracker.generate_report()
        
        self.assertIn('metrics', report)
        self.assertIn('statistics', report)
        self.assertIn('alerts', report)
        
    def test_alert_throttling(self):
        """测试告警节流"""
        alert_data = {
            'type': 'cpu_alert',
            'message': 'CPU usage exceeds threshold',
            'value': 90
        }
        
        # 第一次告警应该发送
        self.assertTrue(self.monitor.should_send_alert(alert_data))
        
        # 冷却期内的告警应该被抑制
        self.assertFalse(self.monitor.should_send_alert(alert_data))
        
    def test_resource_tracking(self):
        """测试资源追踪"""
        self.monitor.track_resources()
        
        resources = self.monitor.get_resource_usage()
        self.assertIn('cpu', resources)
        self.assertIn('memory', resources)
        self.assertIn('disk', resources)
        
    def test_error_tracking(self):
        """测试错误追踪"""
        error = Exception("Test error")
        self.monitor.track_error(error)
        
        errors = self.monitor.get_error_log()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['message'], "Test error")
        
    def test_log_rotation(self):
        """测试日志轮转"""
        for i in range(100):
            self.monitor.log_metric({
                'cpu': i,
                'memory': i
            })
            
        logs = self.monitor.get_logs()
        self.assertLessEqual(len(logs), self.monitor.MAX_LOG_ENTRIES)
        
    def test_metric_aggregation(self):
        """测试指标聚合"""
        metrics = [
            {'cpu': 80, 'memory': 70},
            {'cpu': 85, 'memory': 75},
            {'cpu': 90, 'memory': 80}
        ]
        
        aggregated = self.monitor.aggregate_metrics(metrics)
        
        self.assertIn('cpu_avg', aggregated)
        self.assertIn('memory_avg', aggregated)
        self.assertIn('cpu_max', aggregated)
        self.assertIn('memory_max', aggregated)
        
    def tearDown(self):
        """测试后的清理"""
        self.monitor = None
        self.notifier = None
        self.tracker = None

if __name__ == '__main__':
    unittest.main() 
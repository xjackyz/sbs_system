import unittest
import os
import json
from unittest.mock import Mock, patch
from src.main import SBSSystem, SystemConfig
from src.data.collector import DataCollector
from src.model.analyzer import LLaVAAnalyzer
from src.signal.signal_generator import SignalGenerator
from src.monitor.system_monitor import SystemMonitor

class TestMain(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = SystemConfig(
            run_mode='test',
            debug=True,
            log_level='INFO',
            use_gpu=False,
            num_workers=2,
            data_dir='test_data',
            model_dir='test_models',
            log_dir='test_logs'
        )
        
        # 创建测试目录
        for dir_path in [self.config.data_dir, self.config.model_dir, self.config.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # 模拟各个组件
        self.mock_collector = Mock(spec=DataCollector)
        self.mock_analyzer = Mock(spec=LLaVAAnalyzer)
        self.mock_generator = Mock(spec=SignalGenerator)
        self.mock_monitor = Mock(spec=SystemMonitor)
        
        self.system = SBSSystem(
            config=self.config,
            collector=self.mock_collector,
            analyzer=self.mock_analyzer,
            generator=self.mock_generator,
            monitor=self.mock_monitor
        )
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.system.config.run_mode, 'test')
        self.assertTrue(self.system.config.debug)
        self.assertEqual(self.system.config.log_level, 'INFO')
        
        self.assertIsNotNone(self.system.collector)
        self.assertIsNotNone(self.system.analyzer)
        self.assertIsNotNone(self.system.generator)
        self.assertIsNotNone(self.system.monitor)
        
    def test_system_startup(self):
        """测试系统启动"""
        self.system.startup()
        
        self.mock_collector.initialize.assert_called_once()
        self.mock_analyzer.initialize.assert_called_once()
        self.mock_generator.initialize.assert_called_once()
        self.mock_monitor.start_monitoring.assert_called_once()
        
    def test_system_shutdown(self):
        """测试系统关闭"""
        self.system.shutdown()
        
        self.mock_collector.cleanup.assert_called_once()
        self.mock_analyzer.cleanup.assert_called_once()
        self.mock_generator.cleanup.assert_called_once()
        self.mock_monitor.stop_monitoring.assert_called_once()
        
    @patch('src.main.time.sleep', return_value=None)
    def test_main_loop(self, mock_sleep):
        """测试主循环"""
        # 模拟收集到的数据
        test_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        self.mock_collector.collect_data.return_value = test_data
        
        # 模拟分析结果
        test_analysis = {
            'market_analysis': {
                'trend': 'upward',
                'strength': 0.8
            },
            'sequence_analysis': {
                'type': 'sbs_upward',
                'confidence': 0.85
            }
        }
        self.mock_analyzer.analyze.return_value = test_analysis
        
        # 模拟信号生成
        test_signal = {
            'direction': 'long',
            'entry': {'price': 40000, 'zone': (39900, 40100)},
            'stop_loss': 39500,
            'take_profit': [40500, 41000, 41500],
            'confidence': 0.85
        }
        self.mock_generator.generate_signal.return_value = test_signal
        
        # 运行一次主循环
        self.system.run_once()
        
        # 验证各组件是否被正确调用
        self.mock_collector.collect_data.assert_called_once()
        self.mock_analyzer.analyze.assert_called_once_with(test_data)
        self.mock_generator.generate_signal.assert_called_once_with(test_analysis)
        self.mock_monitor.track_metrics.assert_called()
        
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟数据收集错误
        self.mock_collector.collect_data.side_effect = Exception("Data collection error")
        
        with self.assertLogs(level='ERROR') as log:
            self.system.run_once()
            
        self.assertIn("Data collection error", log.output[0])
        self.mock_monitor.track_error.assert_called()
        
    def test_system_state(self):
        """测试系统状态"""
        state = self.system.get_system_state()
        
        self.assertIn('run_mode', state)
        self.assertIn('components_status', state)
        self.assertIn('last_update', state)
        
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = SystemConfig(
            run_mode='test',
            debug=True,
            log_level='INFO',
            use_gpu=False,
            num_workers=2,
            data_dir='test_data',
            model_dir='test_models',
            log_dir='test_logs'
        )
        self.assertTrue(self.system.validate_config(valid_config))
        
        # 测试无效配置
        invalid_config = SystemConfig(
            run_mode='invalid_mode',
            debug=True,
            log_level='INVALID',
            use_gpu=False,
            num_workers=-1,
            data_dir='',
            model_dir='',
            log_dir=''
        )
        self.assertFalse(self.system.validate_config(invalid_config))
        
    def test_component_health_check(self):
        """测试组件健康检查"""
        health_status = self.system.check_components_health()
        
        self.assertIn('collector', health_status)
        self.assertIn('analyzer', health_status)
        self.assertIn('generator', health_status)
        self.assertIn('monitor', health_status)
        
    def test_system_metrics(self):
        """测试系统指标"""
        metrics = self.system.get_system_metrics()
        
        self.assertIn('processing_time', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('success_rate', metrics)
        
    def test_data_flow(self):
        """测试数据流"""
        test_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        
        # 测试数据处理流程
        processed_data = self.system.process_data(test_data)
        
        self.assertIsNotNone(processed_data)
        self.mock_analyzer.preprocess.assert_called_once()
        self.mock_analyzer.analyze.assert_called_once()
        
    def test_signal_processing(self):
        """测试信号处理"""
        test_signal = {
            'direction': 'long',
            'entry': {'price': 40000, 'zone': (39900, 40100)},
            'stop_loss': 39500,
            'take_profit': [40500, 41000, 41500],
            'confidence': 0.85
        }
        
        # 测试信号处理流程
        processed_signal = self.system.process_signal(test_signal)
        
        self.assertIsNotNone(processed_signal)
        self.mock_monitor.track_signal.assert_called_once()
        
    def test_performance_tracking(self):
        """测试性能追踪"""
        # 运行系统一段时间
        for _ in range(5):
            self.system.run_once()
            
        performance_metrics = self.system.get_performance_metrics()
        
        self.assertIn('average_processing_time', performance_metrics)
        self.assertIn('signal_accuracy', performance_metrics)
        self.assertIn('system_uptime', performance_metrics)
        
    def tearDown(self):
        """测试后的清理"""
        import shutil
        for dir_path in [self.config.data_dir, self.config.model_dir, self.config.log_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                
        self.system = None

if __name__ == '__main__':
    unittest.main() 
import unittest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from src.main import SBSSystem
from src.data.collector import DataCollector
from src.model.llava_analyzer import LLaVAAnalyzer
from src.signal.signal_generator import SignalGenerator
from src.monitor.system_monitor import SystemMonitor

class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.system = SBSSystem()
        cls.loop = asyncio.get_event_loop()
        
    def setUp(self):
        """每个测试用例初始化"""
        self.collector = Mock(spec=DataCollector)
        self.analyzer = Mock(spec=LLaVAAnalyzer)
        self.generator = Mock(spec=SignalGenerator)
        self.monitor = Mock(spec=SystemMonitor)
        
        self.system.collector = self.collector
        self.system.analyzer = self.analyzer
        self.system.generator = self.generator
        self.system.monitor = self.monitor
        
    async def test_full_workflow(self):
        """测试完整工作流程"""
        # 模拟数据收集
        test_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'image_path': 'test.png',
            'timestamp': datetime.now().isoformat()
        }
        self.collector.collect_data.return_value = test_data
        
        # 模拟图表分析
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
        self.analyzer.analyze_chart.return_value = test_analysis
        
        # 模拟信号生成
        test_signal = {
            'direction': 'long',
            'entry': {'price': 40000, 'zone': (39900, 40100)},
            'stop_loss': 39500,
            'take_profit': [40500, 41000, 41500],
            'confidence': 0.85
        }
        self.generator.generate_signal.return_value = test_signal
        
        # 运行系统
        await self.system.run_once()
        
        # 验证组件调用
        self.collector.collect_data.assert_called_once()
        self.analyzer.analyze_chart.assert_called_once()
        self.generator.generate_signal.assert_called_once_with(test_analysis)
        self.monitor.track_metrics.assert_called()
        
    async def test_error_handling(self):
        """测试错误处理"""
        # 模拟数据收集错误
        self.collector.collect_data.side_effect = Exception("数据收集失败")
        
        # 运行系统
        await self.system.run_once()
        
        # 验证错误处理
        self.monitor.track_error.assert_called()
        self.analyzer.analyze_chart.assert_not_called()
        self.generator.generate_signal.assert_not_called()
        
    async def test_data_validation(self):
        """测试数据验证"""
        # 模拟无效数据
        invalid_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'image_path': None,
            'timestamp': None
        }
        self.collector.collect_data.return_value = invalid_data
        
        # 运行系统
        await self.system.run_once()
        
        # 验证数据验证
        self.analyzer.analyze_chart.assert_not_called()
        self.monitor.track_error.assert_called()
        
    async def test_system_recovery(self):
        """测试系统恢复"""
        # 模拟连续错误
        self.collector.collect_data.side_effect = [
            Exception("错误1"),
            Exception("错误2"),
            {'symbol': 'BTCUSDT', 'timeframe': '1h', 'image_path': 'test.png'}
        ]
        
        # 运行系统多次
        for _ in range(3):
            await self.system.run_once()
        
        # 验证系统恢复
        self.assertEqual(self.monitor.track_error.call_count, 2)
        self.assertEqual(self.collector.collect_data.call_count, 3)
        
    async def test_performance_monitoring(self):
        """测试性能监控"""
        # 运行系统
        await self.system.run_once()
        
        # 验证性能监控
        self.monitor.track_metrics.assert_called()
        metrics = self.monitor.track_metrics.call_args[0][0]
        
        self.assertIn('processing_time', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('success_rate', metrics)
        
    def test_component_initialization(self):
        """测试组件初始化"""
        system = SBSSystem()
        
        self.assertIsNotNone(system.collector)
        self.assertIsNotNone(system.analyzer)
        self.assertIsNotNone(system.generator)
        self.assertIsNotNone(system.monitor)
        
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = {
            'preprocessing': {'target_size': (224, 224)},
            'model': {'confidence_threshold': 0.8},
            'signal': {'min_confidence': 0.75},
            'discord': {'webhook_url': 'https://...'}
        }
        
        self.assertTrue(self.system.validate_config(valid_config))
        
        # 测试无效配置
        invalid_config = {
            'preprocessing': {},
            'model': {},
            'signal': {},
            'discord': {}
        }
        
        self.assertFalse(self.system.validate_config(invalid_config))
        
    def tearDown(self):
        """测试用例清理"""
        self.collector = None
        self.analyzer = None
        self.generator = None
        self.monitor = None
        
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.loop.close()

def run_async_test(coro):
    """运行异步测试的辅助函数"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

if __name__ == '__main__':
    unittest.main() 
import unittest
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch
from src.data.collector import DataCollector, CollectorConfig
from src.data.tradingview import TradingViewClient

class TestDataCollector(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = CollectorConfig(
            screenshot_interval=300,
            tradingview_api_url="https://api.tradingview.com",
            tradingview_api_token="test_token",
            backup_enabled=True,
            backup_interval=3600,
            compression_enabled=True,
            max_retries=3,
            retry_delay=5,
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1h', '4h', '1d'],
            indicators=['RSI', 'MACD', 'BB']
        )
        
        self.mock_tv_client = Mock(spec=TradingViewClient)
        self.collector = DataCollector(self.config, tv_client=self.mock_tv_client)
        
        # 创建测试目录
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.collector.config.screenshot_interval, 300)
        self.assertEqual(self.collector.config.symbols, ['BTCUSDT', 'ETHUSDT'])
        self.assertTrue(self.collector.config.backup_enabled)
        
    @patch('src.data.collector.requests.get')
    def test_fetch_chart_data(self, mock_get):
        """测试图表数据获取"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'data': {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500},
                    {'time': '2024-01-01 01:00:00', 'open': 40500, 'high': 42000, 'low': 40000, 'close': 41000}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        data = self.collector.fetch_chart_data('BTCUSDT', '1h')
        
        self.assertIsNotNone(data)
        self.assertEqual(data['symbol'], 'BTCUSDT')
        self.assertEqual(len(data['data']['candles']), 2)
        
    def test_save_chart_data(self):
        """测试图表数据保存"""
        test_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        
        filename = self.collector.save_chart_data(test_data, self.test_dir)
        
        self.assertTrue(os.path.exists(filename))
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data['symbol'], 'BTCUSDT')
        
    def test_compress_data(self):
        """测试数据压缩"""
        test_file = os.path.join(self.test_dir, 'test.json')
        with open(test_file, 'w') as f:
            json.dump({'test': 'data'}, f)
            
        compressed_file = self.collector.compress_data(test_file)
        
        self.assertTrue(os.path.exists(compressed_file))
        self.assertTrue(compressed_file.endswith('.gz'))
        
    def test_backup_data(self):
        """测试数据备份"""
        test_file = os.path.join(self.test_dir, 'test.json')
        with open(test_file, 'w') as f:
            json.dump({'test': 'data'}, f)
            
        backup_file = self.collector.backup_data(test_file)
        
        self.assertTrue(os.path.exists(backup_file))
        self.assertIn('backup', backup_file)
        
    @patch('src.data.collector.time.sleep', return_value=None)
    def test_retry_mechanism(self, mock_sleep):
        """测试重试机制"""
        def failing_function():
            raise Exception("API Error")
            
        with self.assertRaises(Exception):
            self.collector.retry_with_backoff(failing_function)
            
        self.assertEqual(mock_sleep.call_count, self.config.max_retries)
        
    def test_validate_data(self):
        """测试数据验证"""
        valid_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        
        self.assertTrue(self.collector.validate_data(valid_data))
        
        invalid_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {}
        }
        
        self.assertFalse(self.collector.validate_data(invalid_data))
        
    def test_process_indicators(self):
        """测试指标处理"""
        candle_data = [
            {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500},
            {'time': '2024-01-01 01:00:00', 'open': 40500, 'high': 42000, 'low': 40000, 'close': 41000}
        ]
        
        processed_data = self.collector.process_indicators(candle_data)
        
        self.assertIn('RSI', processed_data[0])
        self.assertIn('MACD', processed_data[0])
        self.assertIn('BB', processed_data[0])
        
    def test_data_integrity(self):
        """测试数据完整性"""
        test_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': 40000, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        
        integrity_check = self.collector.check_data_integrity(test_data)
        self.assertTrue(integrity_check)
        
        # 测试缺失数据
        incomplete_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'data': {
                'candles': [
                    {'time': '2024-01-01 00:00:00', 'open': None, 'high': 41000, 'low': 39000, 'close': 40500}
                ]
            }
        }
        
        integrity_check = self.collector.check_data_integrity(incomplete_data)
        self.assertFalse(integrity_check)
        
    def test_error_logging(self):
        """测试错误日志记录"""
        with self.assertLogs(level='ERROR') as log:
            self.collector.log_error("Test error message")
            
        self.assertIn("Test error message", log.output[0])
        
    def tearDown(self):
        """测试后的清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.collector = None

if __name__ == '__main__':
    unittest.main() 
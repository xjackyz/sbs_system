import unittest
import os
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.main import SBSSystem
from src.utils.logger import setup_logger

logger = setup_logger('e2e_test')

class TestSystemE2E(unittest.TestCase):
    """系统端到端测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建测试目录
        cls.test_dirs = ['test_data', 'test_models', 'test_logs']
        for dir_path in cls.test_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        # 创建测试数据
        cls._create_test_data()
        
        # 初始化系统
        cls.system = SBSSystem()
        cls.loop = asyncio.get_event_loop()
        
    @classmethod
    def _create_test_data(cls):
        """创建测试数据"""
        # 创建模拟行情数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(40000, 1000, len(dates)),
            'high': np.random.normal(41000, 1000, len(dates)),
            'low': np.random.normal(39000, 1000, len(dates)),
            'close': np.random.normal(40500, 1000, len(dates)),
            'volume': np.random.normal(1000, 100, len(dates))
        })
        
        # 保存数据
        data.to_csv('test_data/BTCUSDT_1h.csv', index=False)
        
    async def test_system_startup(self):
        """测试系统启动"""
        # 启动系统
        await self.system.startup()
        
        # 验证系统状态
        system_state = self.system.get_system_state()
        self.assertEqual(system_state['status'], 'running')
        self.assertTrue(all(c['status'] == 'ready' for c in system_state['components'].values()))
        
    async def test_data_collection_to_signal(self):
        """测试从数据收集到信号生成的完整流程"""
        # 运行系统一个完整周期
        await self.system.run_once()
        
        # 获取系统指标
        metrics = self.system.get_system_metrics()
        
        # 验证关键指标
        self.assertGreater(metrics['success_rate'], 0.8)
        self.assertLess(metrics['error_rate'], 0.2)
        self.assertGreater(metrics['data_quality_score'], 0.7)
        
    async def test_error_recovery(self):
        """测试错误恢复机制"""
        # 模拟系统崩溃
        await self.system.simulate_crash()
        
        # 等待恢复
        await asyncio.sleep(5)
        
        # 验证系统恢复
        system_state = self.system.get_system_state()
        self.assertEqual(system_state['status'], 'running')
        
    async def test_data_persistence(self):
        """测试数据持久化"""
        # 生成测试信号
        test_signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            'direction': 'long',
            'entry': 40000,
            'stop_loss': 39500,
            'take_profit': [40500, 41000]
        }
        
        # 保存信号
        await self.system.save_signal(test_signal)
        
        # 验证信号已保存
        saved_signals = await self.system.load_signals(
            start_time=datetime.now() - timedelta(hours=1)
        )
        self.assertGreater(len(saved_signals), 0)
        
    async def test_monitoring_system(self):
        """测试监控系统"""
        # 运行系统一段时间
        for _ in range(5):
            await self.system.run_once()
            await asyncio.sleep(1)
        
        # 获取监控数据
        monitoring_data = self.system.get_monitoring_data()
        
        # 验证监控指标
        self.assertIn('system_metrics', monitoring_data)
        self.assertIn('component_status', monitoring_data)
        self.assertIn('error_logs', monitoring_data)
        
    async def test_data_validation(self):
        """测试数据验证机制"""
        # 准备测试数据
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [40000],
            'high': [41000],
            'low': [39000],
            'close': [40500],
            'volume': [1000]
        })
        
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [np.nan],
            'high': [np.inf],
            'low': [-1000],
            'close': [0],
            'volume': [-100]
        })
        
        # 验证有效数据
        validation_result = await self.system.validate_data(valid_data)
        self.assertTrue(validation_result['is_valid'])
        
        # 验证无效数据
        validation_result = await self.system.validate_data(invalid_data)
        self.assertFalse(validation_result['is_valid'])
        
    async def test_performance_under_load(self):
        """测试系统负载性能"""
        # 模拟高负载
        tasks = []
        for _ in range(10):
            tasks.append(self.system.run_once())
        
        # 并发执行
        await asyncio.gather(*tasks)
        
        # 验证系统性能
        performance_metrics = self.system.get_performance_metrics()
        self.assertLess(performance_metrics['average_response_time'], 1.0)
        self.assertLess(performance_metrics['error_rate'], 0.1)
        
    def test_configuration_management(self):
        """测试配置管理"""
        # 测试配置加载
        config = self.system.load_config()
        self.assertIsNotNone(config)
        
        # 测试配置验证
        validation_result = self.system.validate_config(config)
        self.assertTrue(validation_result['is_valid'])
        
        # 测试配置更新
        updated_config = config.copy()
        updated_config['scan_interval'] = 30
        self.system.update_config(updated_config)
        
        new_config = self.system.load_config()
        self.assertEqual(new_config['scan_interval'], 30)
        
    def tearDown(self):
        """测试用例清理"""
        pass
        
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        # 清理测试目录
        for dir_path in cls.test_dirs:
            if os.path.exists(dir_path):
                import shutil
                shutil.rmtree(dir_path)
        
        # 关闭事件循环
        cls.loop.close()

def run_async_test(coro):
    """运行异步测试的辅助函数"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

if __name__ == '__main__':
    unittest.main() 
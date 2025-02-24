import unittest
import asyncio
import time
import psutil
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.main import SBSSystem
from src.utils.logger import setup_logger
from src.utils.memory_monitor import MemoryMonitor

logger = setup_logger('performance_test')

class TestSystemPerformance(unittest.TestCase):
    """系统性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.system = SBSSystem()
        cls.memory_monitor = MemoryMonitor()
        cls.loop = asyncio.get_event_loop()
        
        # 创建测试数据
        cls._create_test_data()
        
    @classmethod
    def _create_test_data(cls):
        """创建性能测试数据"""
        # 创建大量测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
        cls.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(40000, 1000, len(dates)),
            'high': np.random.normal(41000, 1000, len(dates)),
            'low': np.random.normal(39000, 1000, len(dates)),
            'close': np.random.normal(40500, 1000, len(dates)),
            'volume': np.random.normal(1000, 100, len(dates))
        })
        
    async def test_response_time(self):
        """测试系统响应时间"""
        response_times = []
        
        for _ in range(100):
            start_time = time.time()
            await self.system.run_once()
            response_time = time.time() - start_time
            response_times.append(response_time)
            
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        logger.info(f"平均响应时间: {avg_response_time:.3f}秒")
        logger.info(f"95%响应时间: {p95_response_time:.3f}秒")
        
        # 验证性能指标
        self.assertLess(avg_response_time, 0.5)  # 平均响应时间小于0.5秒
        self.assertLess(p95_response_time, 1.0)  # 95%响应时间小于1秒
        
    async def test_memory_usage(self):
        """测试内存使用"""
        # 开始监控内存
        self.memory_monitor.start()
        
        # 运行系统一段时间
        for _ in range(50):
            await self.system.run_once()
            
        # 获取内存使用情况
        memory_stats = self.memory_monitor.get_stats()
        
        logger.info(f"平均内存使用: {memory_stats['avg_memory_usage']:.2f}MB")
        logger.info(f"峰值内存使用: {memory_stats['peak_memory_usage']:.2f}MB")
        
        # 验证内存使用
        self.assertLess(memory_stats['avg_memory_usage'], 1024)  # 平均内存使用小于1GB
        self.assertLess(memory_stats['peak_memory_usage'], 2048)  # 峰值内存使用小于2GB
        
    async def test_concurrent_processing(self):
        """测试并发处理能力"""
        # 创建并发任务
        num_tasks = 20
        tasks = []
        
        start_time = time.time()
        
        for _ in range(num_tasks):
            tasks.append(self.system.run_once())
            
        # 并发执行
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        throughput = num_tasks / total_time
        
        logger.info(f"并发吞吐量: {throughput:.2f}请求/秒")
        
        # 验证并发性能
        self.assertGreater(throughput, 5.0)  # 吞吐量大于5请求/秒
        
    def test_cpu_usage(self):
        """测试CPU使用率"""
        cpu_usage = []
        
        def monitor_cpu():
            return psutil.cpu_percent(interval=1)
        
        # 在后台运行系统
        with ThreadPoolExecutor() as executor:
            system_future = executor.submit(
                self.loop.run_until_complete,
                self._run_system_background()
            )
            
            # 监控CPU使用率
            for _ in range(30):
                cpu_usage.append(monitor_cpu())
                
            system_future.result()
            
        avg_cpu = np.mean(cpu_usage)
        max_cpu = np.max(cpu_usage)
        
        logger.info(f"平均CPU使用率: {avg_cpu:.2f}%")
        logger.info(f"最大CPU使用率: {max_cpu:.2f}%")
        
        # 验证CPU使用率
        self.assertLess(avg_cpu, 70)  # 平均CPU使用率小于70%
        self.assertLess(max_cpu, 90)  # 最大CPU使用率小于90%
        
    async def test_data_processing_speed(self):
        """测试数据处理速度"""
        chunk_size = 1000
        num_chunks = len(self.test_data) // chunk_size
        processing_times = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = self.test_data.iloc[start_idx:end_idx]
            
            start_time = time.time()
            await self.system.process_data(chunk)
            processing_times.append(time.time() - start_time)
            
        avg_processing_time = np.mean(processing_times)
        processing_rate = chunk_size / avg_processing_time
        
        logger.info(f"数据处理速率: {processing_rate:.2f}条/秒")
        
        # 验证处理速度
        self.assertGreater(processing_rate, 100)  # 处理速率大于100条/秒
        
    async def test_long_running_stability(self):
        """测试长时间运行稳定性"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)
        error_count = 0
        total_runs = 0
        
        while datetime.now() < end_time:
            try:
                await self.system.run_once()
                total_runs += 1
            except Exception as e:
                error_count += 1
                logger.error(f"运行错误: {e}")
                
        error_rate = error_count / total_runs if total_runs > 0 else 1
        
        logger.info(f"总运行次数: {total_runs}")
        logger.info(f"错误率: {error_rate:.2%}")
        
        # 验证稳定性
        self.assertLess(error_rate, 0.01)  # 错误率小于1%
        
    async def _run_system_background(self):
        """后台运行系统"""
        for _ in range(10):
            await self.system.run_once()
            await asyncio.sleep(0.1)
            
    def tearDown(self):
        """测试用例清理"""
        self.memory_monitor.stop()
        
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
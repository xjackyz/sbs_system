"""
测试运行器模块
"""
import importlib
import inspect
import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time

from src.utils.logger import setup_logger
from src.tests.test_collector import TestCollector

logger = setup_logger('test_runner')

class TestRunner:
    """测试运行器类"""
    
    def __init__(self):
        self.collector = TestCollector()
        self.test_dir = Path('tests')
        self.current_suite = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试并收集结果"""
        logger.info("开始运行所有测试...")
        start_time = time.time()
        
        test_suites = [
            'preprocessing_tests',
            'model_tests',
            'signal_tests',
            'notification_tests',
            'integration_tests'
        ]
        
        total_tests = 0
        for suite in test_suites:
            logger.info(f"\n运行测试套件: {suite}")
            suite_count = self.run_test_suite(suite)
            total_tests += suite_count
            
        duration = time.time() - start_time
        logger.info(f"\n所有测试完成，总耗时: {duration:.2f}秒")
        
        # 生成并保存报告
        report = self.collector.generate_report()
        self.collector.save_report()
        self.collector.print_summary()
        
        return report
        
    def run_test_suite(self, suite_name: str) -> int:
        """运行特定测试套件"""
        try:
            suite = self.load_test_suite(suite_name)
            if not suite:
                logger.warning(f"测试套件为空: {suite_name}")
                return 0
                
            logger.info(f"加载测试用例: {len(suite)}个")
            self.current_suite = suite_name
            self.collector.collect_results(suite)
            
            return len(suite)
            
        except Exception as e:
            logger.error(f"运行测试套件{suite_name}时出错: {e}")
            return 0
            
    def load_test_suite(self, suite_name: str) -> List:
        """加载测试套件"""
        try:
            # 构建模块路径
            module_path = f"tests.{suite_name}"
            
            # 导入模块
            module = importlib.import_module(module_path)
            
            # 获取所有测试用例
            test_cases = []
            for name, obj in inspect.getmembers(module):
                # 检查是否是测试方法
                if name.startswith('test_') and callable(obj):
                    test_cases.append(obj)
                # 检查是否是测试类
                elif inspect.isclass(obj) and name.startswith('Test'):
                    instance = obj()
                    for method_name, method in inspect.getmembers(instance):
                        if method_name.startswith('test_') and callable(method):
                            test_cases.append(method)
                            
            return test_cases
            
        except ImportError as e:
            logger.error(f"导入测试套件{suite_name}失败: {e}")
            return []
        except Exception as e:
            logger.error(f"加载测试套件{suite_name}时出错: {e}")
            return []
            
    def run_specific_tests(self, test_names: List[str]) -> Dict[str, Any]:
        """运行指定的测试"""
        logger.info(f"运行指定的测试: {test_names}")
        
        test_cases = []
        for suite_name in self._get_all_test_suites():
            suite = self.load_test_suite(suite_name)
            for test in suite:
                if test.__name__ in test_names:
                    test_cases.append(test)
                    
        if not test_cases:
            logger.warning("未找到指定的测试用例")
            return {}
            
        self.collector.collect_results(test_cases)
        return self.collector.generate_report()
        
    def _get_all_test_suites(self) -> List[str]:
        """获取所有测试套件名称"""
        suites = []
        test_dir = Path('tests')
        
        if not test_dir.exists():
            return suites
            
        for item in test_dir.iterdir():
            if item.is_file() and item.name.startswith('test_') and item.name.endswith('.py'):
                suite_name = item.stem
                suites.append(suite_name)
                
        return suites
        
    def cleanup(self):
        """清理资源"""
        self.collector.reset()
        self.current_suite = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 
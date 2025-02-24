"""
测试结果收集器模块
"""
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from src.utils.logger import setup_logger

logger = setup_logger('test_collector')

class TestCollector:
    """测试结果收集器类"""
    
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'errors': [],
            'skipped': []
        }
        self.error_details = {}
        self.start_time = None
        self.end_time = None
        
    def run_test(self, test) -> Dict[str, Any]:
        """运行单个测试"""
        try:
            # 记录开始时间
            test_start = datetime.now()
            
            # 运行测试
            if hasattr(test, 'skip') and test.skip:
                result = {
                    'name': test.__name__,
                    'status': 'skipped',
                    'message': getattr(test, 'skip_reason', 'Test was skipped'),
                    'duration': 0
                }
            else:
                # 执行测试
                test_result = test()
                duration = (datetime.now() - test_start).total_seconds()
                
                result = {
                    'name': test.__name__,
                    'status': 'passed' if test_result else 'failed',
                    'duration': duration
                }
                
                if not test_result:
                    result['message'] = getattr(test, 'fail_message', 'Test failed')
                    
            return result
            
        except Exception as e:
            duration = (datetime.now() - test_start).total_seconds()
            return {
                'name': test.__name__,
                'status': 'error',
                'message': str(e),
                'duration': duration
            }
            
    def categorize_result(self, result: Dict[str, Any]):
        """对测试结果进行分类"""
        status = result['status']
        test_name = result['name']
        
        # 添加到对应类别
        self.test_results[status].append(test_name)
        
        # 如果是失败或错误，记录详细信息
        if status in ['failed', 'error']:
            self.error_details[test_name] = {
                'message': result.get('message', ''),
                'duration': result.get('duration', 0)
            }
            
    def collect_results(self, test_suite):
        """收集所有测试结果"""
        self.start_time = datetime.now()
        
        for test in test_suite:
            try:
                result = self.run_test(test)
                self.categorize_result(result)
            except Exception as e:
                self.error_details[test.__name__] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
        self.end_time = datetime.now()
        
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'summary': {
                'total_tests': len(self.test_results['passed']) + 
                             len(self.test_results['failed']) +
                             len(self.test_results['errors']) +
                             len(self.test_results['skipped']),
                'passed': len(self.test_results['passed']),
                'failed': len(self.test_results['failed']),
                'errors': len(self.test_results['errors']),
                'skipped': len(self.test_results['skipped']),
                'duration': total_duration
            },
            'failed_tests': self.error_details,
            'timestamp': datetime.now().isoformat()
        }
        
    def save_report(self, report_dir: str = 'test_reports'):
        """保存测试报告"""
        try:
            # 创建报告目录
            report_dir = Path(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = report_dir / f"test_report_{timestamp}.json"
            
            # 保存报告
            report = self.generate_report()
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"测试报告已保存至: {report_path}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
            
    def print_summary(self):
        """打印测试摘要"""
        report = self.generate_report()
        summary = report['summary']
        
        print("\n=== 测试结果摘要 ===")
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['passed']}")
        print(f"失败: {summary['failed']}")
        print(f"错误: {summary['errors']}")
        print(f"跳过: {summary['skipped']}")
        print(f"总耗时: {summary['duration']:.2f}秒")
        
        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\n=== 失败详情 ===")
            for test_name, details in report['failed_tests'].items():
                print(f"\n{test_name}:")
                print(f"  消息: {details.get('message', 'N/A')}")
                print(f"  耗时: {details.get('duration', 0):.2f}秒")
                if 'traceback' in details:
                    print(f"  错误追踪:\n{details['traceback']}")
                    
    def reset(self):
        """重置收集器状态"""
        self.test_results = {
            'passed': [],
            'failed': [],
            'errors': [],
            'skipped': []
        }
        self.error_details = {}
        self.start_time = None
        self.end_time = None 
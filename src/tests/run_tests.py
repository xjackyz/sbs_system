"""
测试执行器模块，用于运行测试并分析结果
"""
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.tests.test_runner import TestRunner
from src.tests.error_analyzer import ErrorAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger('test_executor')

class TestExecutor:
    """测试执行器类，负责运行测试并分析结果"""
    
    def __init__(self):
        """初始化测试执行器"""
        self.runner = TestRunner()
        self.analyzer = ErrorAnalyzer()
        self.error_collection = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        self.fix_suggestions = {}
        
    def execute_all_tests(self, save_report: bool = True):
        """
        执行所有测试并生成报告
        
        Args:
            save_report: 是否保存报告到文件
        """
        try:
            # 运行所有测试套件
            logger.info("开始执行所有测试套件...")
            all_test_reports = self._collect_all_test_results()
            
            # 汇总所有错误
            total_summary = self._aggregate_test_results(all_test_reports)
            
            # 如果有错误，分析并生成修复建议
            if total_summary['failed'] > 0 or total_summary['errors'] > 0:
                self._analyze_all_errors(all_test_reports)
                self.display_results(total_summary, self.error_collection)
                
                if save_report:
                    self._save_error_report()
            else:
                logger.info("所有测试通过！")
                print("\n✅ 所有测试通过！")
    
        except Exception as e:
            logger.error(f"执行测试过程中发生错误: {e}")
            raise
            
    def _collect_all_test_results(self) -> List[Dict]:
        """收集所有测试结果"""
        test_suites = self.runner._get_all_test_suites()
        all_reports = []
        
        for suite in test_suites:
            logger.info(f"执行测试套件: {suite}")
            report = self.runner.run_test_suite(suite)
            all_reports.append({
                'suite_name': suite,
                'report': report
            })
            
        return all_reports
        
    def _aggregate_test_results(self, all_reports: List[Dict]) -> Dict:
        """汇总所有测试结果"""
        total_summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        }
        
        for report in all_reports:
            summary = report['report']['summary']
            for key in total_summary:
                total_summary[key] += summary[key]
                
        return total_summary
        
    def _analyze_all_errors(self, all_reports: List[Dict]):
        """分析所有错误并生成修复建议"""
        for report in all_reports:
            suite_name = report['suite_name']
            test_report = report['report']
            
            if test_report['failed_tests'] or test_report['error_tests']:
                fix_suggestions = self.analyzer.analyze_errors(test_report)
                
                # 按优先级收集错误
                for priority in ['high', 'medium', 'low']:
                    priority_key = f'{priority}_priority'
                    for fix in fix_suggestions[priority_key]:
                        fix['suite'] = suite_name
                        self.error_collection[priority_key].append(fix)
    
    def display_results(self, total_summary: Dict[str, int], error_collection: Dict[str, List]):
        """显示测试结果和修复建议"""
        # 显示总体测试摘要
        print("\n=== 测试结果总览 ===")
        print(f"总测试数: {total_summary['total_tests']}")
        print(f"通过: {total_summary['passed']} ✅")
        print(f"失败: {total_summary['failed']} ❌")
        print(f"错误: {total_summary['errors']} 🚫")
        print(f"跳过: {total_summary['skipped']} ⏭️")
        
        # 显示所有错误的汇总
        print("\n=== 错误汇总 ===")
        for priority in ['high', 'medium', 'low']:
            priority_errors = error_collection[f'{priority}_priority']
            if priority_errors:
                print(f"\n{priority.upper()} 优先级错误 ({len(priority_errors)}个):")
                for error in priority_errors:
                    print(f"\n测试套件: {error['suite']}")
                    print(f"测试: {error['test']}")
                    print(f"错误类型: {error['error_type']}")
                    print(f"错误信息: {error['error']}")
        
        # 显示修复建议
        print("\n=== 批量修复建议 ===")
        for priority in ['high', 'medium', 'low']:
            priority_errors = error_collection[f'{priority}_priority']
            if priority_errors:
                print(f"\n{priority.upper()} 优先级修复:")
                for error in priority_errors:
                    print(f"\n{error['test']}:")
                    print("建议修复:")
                    for solution in error['suggested_fix']['solutions']:
                        print(f"- {solution}")
                    if error['code_example']:
                        print("示例代码:")
                        print(error['code_example'])
                        
    def _save_error_report(self):
        """保存错误报告到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("test_reports")
            report_dir.mkdir(exist_ok=True)
            
            report_path = report_dir / f"error_report_{timestamp}.json"
            
            report_data = {
                'timestamp': timestamp,
                'error_collection': self.error_collection,
                'fix_suggestions': self.fix_suggestions
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"错误报告已保存至: {report_path}")
            print(f"\n📝 详细错误报告已保存至: {report_path}")
            
        except Exception as e:
            logger.error(f"保存错误报告时发生错误: {e}")
    
    def execute_specific_tests(self, test_names: list):
        """执行指定的测试"""
        try:
            logger.info(f"开始执行指定测试: {test_names}")
            test_report = self.runner.run_specific_tests(test_names)
            
            if test_report['summary']['failed'] > 0 or test_report['summary']['errors'] > 0:
                self._analyze_all_errors([{'suite_name': 'specific', 'report': test_report}])
                self.display_results(test_report['summary'], self.error_collection)
                self._save_error_report()
            else:
                logger.info("所有指定测试通过！")
                print("\n✅ 所有指定测试通过！")
                
        except Exception as e:
            logger.error(f"执行指定测试过程中发生错误: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        try:
            self.runner.cleanup()
            logger.info("测试执行器资源已清理")
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

def main():
    """主函数"""
    executor = TestExecutor()
    try:
        executor.execute_all_tests()
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(1)
    finally:
        executor.cleanup()

if __name__ == '__main__':
    main() 
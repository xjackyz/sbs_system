"""
错误分析器模块
"""
import re
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger('error_analyzer')

class ErrorAnalyzer:
    """错误分析器类"""
    
    def __init__(self):
        self.common_fixes = self.load_common_fixes()
        self.error_patterns = self.load_error_patterns()
        self.fix_history = []
        
    def load_common_fixes(self) -> Dict[str, Any]:
        """加载常见错误修复方案"""
        try:
            fixes_path = Path('config/common_fixes.json')
            if fixes_path.exists():
                with open(fixes_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 默认修复方案
            return {
                'assertion_error': {
                    'description': '断言失败',
                    'common_causes': [
                        '预期值与实际值不匹配',
                        '数据类型不匹配',
                        '精度误差'
                    ],
                    'solutions': [
                        '检查预期值计算逻辑',
                        '确认数据类型转换',
                        '考虑使用近似比较'
                    ]
                },
                'import_error': {
                    'description': '导入错误',
                    'common_causes': [
                        '模块路径错误',
                        '依赖包未安装',
                        'Python路径配置错误'
                    ],
                    'solutions': [
                        '检查import语句',
                        '安装缺失的依赖',
                        '配置PYTHONPATH'
                    ]
                },
                'type_error': {
                    'description': '类型错误',
                    'common_causes': [
                        '参数类型不匹配',
                        '返回值类型错误',
                        '类型转换失败'
                    ],
                    'solutions': [
                        '添加类型检查',
                        '确保类型转换',
                        '更新类型注解'
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"加载常见修复方案失败: {e}")
            return {}
            
    def load_error_patterns(self) -> Dict[str, str]:
        """加载错误模式匹配规则"""
        return {
            'assertion_error': r'AssertionError:.*',
            'import_error': r'ImportError:.*|ModuleNotFoundError:.*',
            'type_error': r'TypeError:.*',
            'attribute_error': r'AttributeError:.*',
            'index_error': r'IndexError:.*',
            'key_error': r'KeyError:.*',
            'value_error': r'ValueError:.*',
            'runtime_error': r'RuntimeError:.*'
        }
        
    def analyze_errors(self, test_report: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """分析错误并生成修复建议"""
        logger.info("开始分析测试错误...")
        
        fixes = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for test_name, error_info in test_report['failed_tests'].items():
            try:
                fix = self.generate_fix_suggestion(test_name, error_info)
                priority = self.determine_priority(error_info)
                fixes[f'{priority}_priority'].append(fix)
                
                # 记录修复历史
                self.fix_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'test_name': test_name,
                    'error_type': self.identify_error_type(error_info),
                    'fix': fix
                })
                
            except Exception as e:
                logger.error(f"分析错误时出现异常: {e}")
                
        return fixes
        
    def generate_fix_suggestion(self, test_name: str, error_info: Dict) -> Dict[str, Any]:
        """生成具体的修复建议"""
        error_type = self.identify_error_type(error_info)
        solution = self.find_solution(error_info)
        
        return {
            'test': test_name,
            'error': error_info.get('error', ''),
            'error_type': error_type,
            'suggested_fix': solution,
            'code_example': self.generate_code_example(error_info),
            'reference_docs': self.find_reference_docs(error_type),
            'similar_cases': self.find_similar_cases(error_info)
        }
        
    def determine_priority(self, error_info: Dict) -> str:
        """确定错误修复的优先级"""
        error_type = self.identify_error_type(error_info)
        error_message = error_info.get('error', '')
        
        # 高优先级错误
        if any([
            'ImportError' in error_message,
            'ModuleNotFoundError' in error_message,
            'SyntaxError' in error_message
        ]):
            return 'high'
            
        # 中优先级错误
        if any([
            'AssertionError' in error_message,
            'TypeError' in error_message,
            'ValueError' in error_message
        ]):
            return 'medium'
            
        # 低优先级错误
        return 'low'
        
    def identify_error_type(self, error_info: Dict) -> str:
        """识别错误类型"""
        error_message = error_info.get('error', '')
        
        for error_type, pattern in self.error_patterns.items():
            if re.match(pattern, error_message):
                return error_type
                
        return 'unknown_error'
        
    def find_solution(self, error_info: Dict) -> Dict[str, Any]:
        """查找解决方案"""
        error_type = self.identify_error_type(error_info)
        
        if error_type in self.common_fixes:
            fix_info = self.common_fixes[error_type]
            return {
                'description': fix_info['description'],
                'causes': fix_info['common_causes'],
                'solutions': fix_info['solutions']
            }
            
        return {
            'description': '未知错误',
            'causes': ['需要进一步分析'],
            'solutions': ['请查看错误详情并进行调试']
        }
        
    def generate_code_example(self, error_info: Dict) -> Optional[str]:
        """生成示例代码"""
        error_type = self.identify_error_type(error_info)
        
        # 示例代码模板
        code_templates = {
            'assertion_error': '''
def test_example():
    expected = calculate_expected_value()
    actual = get_actual_value()
    assert abs(expected - actual) < 1e-6, "Values should be approximately equal"
''',
            'import_error': '''
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from module import required_function
''',
            'type_error': '''
from typing import List, Dict, Any

def example_function(param: List[int]) -> Dict[str, Any]:
    # 添加类型检查
    if not isinstance(param, list):
        raise TypeError("Expected list type for param")
    return {"result": sum(param)}
'''
        }
        
        return code_templates.get(error_type)
        
    def find_reference_docs(self, error_type: str) -> List[str]:
        """查找相关文档引用"""
        docs = {
            'assertion_error': [
                'https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertAlmostEqual',
                'https://docs.pytest.org/en/stable/how-to/assertions.html'
            ],
            'import_error': [
                'https://docs.python.org/3/tutorial/modules.html',
                'https://docs.python.org/3/command-line.html#envvar-PYTHONPATH'
            ],
            'type_error': [
                'https://docs.python.org/3/library/typing.html',
                'https://mypy.readthedocs.io/en/stable/'
            ]
        }
        
        return docs.get(error_type, [])
        
    def find_similar_cases(self, error_info: Dict) -> List[Dict[str, Any]]:
        """查找类似的错误案例"""
        error_type = self.identify_error_type(error_info)
        
        similar_cases = []
        for case in self.fix_history:
            if case['error_type'] == error_type:
                similar_cases.append({
                    'test_name': case['test_name'],
                    'fix': case['fix']
                })
                
        return similar_cases[-5:]  # 返回最近的5个类似案例
        
    def save_analysis(self, analysis: Dict[str, Any], filename: Optional[str] = None):
        """保存分析结果"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"error_analysis_{timestamp}.json"
                
            report_dir = Path('test_reports/analysis')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = report_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
                
            logger.info(f"错误分析报告已保存至: {report_path}")
            
        except Exception as e:
            logger.error(f"保存错误分析报告失败: {e}")
            
    def print_analysis(self, analysis: Dict[str, List[Dict]]):
        """打印分析结果"""
        print("\n=== 错误分析报告 ===")
        
        for priority in ['high', 'medium', 'low']:
            fixes = analysis[f'{priority}_priority']
            if fixes:
                print(f"\n{priority.upper()}优先级错误:")
                for fix in fixes:
                    print(f"\n测试: {fix['test']}")
                    print(f"错误类型: {fix['error_type']}")
                    print(f"错误信息: {fix['error']}")
                    print("\n建议修复:")
                    for solution in fix['suggested_fix']['solutions']:
                        print(f"- {solution}")
                        
    def cleanup(self):
        """清理资源"""
        self.fix_history = [] 
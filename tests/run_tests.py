import unittest
import sys
import os
import logging
from datetime import datetime

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

def run_tests():
    """运行所有测试用例"""
    # 获取测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 发现所有测试用例
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 创建测试结果收集器
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 运行测试
    logging.info("开始运行测试...")
    result = runner.run(suite)
    
    # 输出测试结果统计
    logging.info("\n测试结果统计:")
    logging.info(f"运行测试用例总数: {result.testsRun}")
    logging.info(f"成功用例数: {result.testsRun - len(result.failures) - len(result.errors)}")
    logging.info(f"失败用例数: {len(result.failures)}")
    logging.info(f"错误用例数: {len(result.errors)}")
    
    # 输出失败的测试用例详情
    if result.failures:
        logging.error("\n失败的测试用例:")
        for failure in result.failures:
            logging.error(f"\n{failure[0]}")
            logging.error(f"原因: {failure[1]}")
            
    # 输出错误的测试用例详情
    if result.errors:
        logging.error("\n错误的测试用例:")
        for error in result.errors:
            logging.error(f"\n{error[0]}")
            logging.error(f"原因: {error[1]}")
            
    # 返回测试是否全部通过
    return len(result.failures) == 0 and len(result.errors) == 0

def setup_test_environment():
    """设置测试环境"""
    # 创建必要的测试目录
    test_dirs = ['test_data', 'test_models', 'test_logs']
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    # 设置测试环境变量
    os.environ['SBS_TEST_MODE'] = 'true'
    os.environ['SBS_CONFIG_PATH'] = 'test_config.py'
    
    logging.info("测试环境设置完成")

def cleanup_test_environment():
    """清理测试环境"""
    # 清理测试目录
    test_dirs = ['test_data', 'test_models', 'test_logs']
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            
    # 清理环境变量
    os.environ.pop('SBS_TEST_MODE', None)
    os.environ.pop('SBS_CONFIG_PATH', None)
    
    logging.info("测试环境清理完成")

if __name__ == '__main__':
    try:
        # 设置测试环境
        setup_test_environment()
        
        # 运行测试
        success = run_tests()
        
        # 根据测试结果设置退出码
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logging.error(f"测试运行过程中发生错误: {str(e)}")
        sys.exit(1)
        
    finally:
        # 清理测试环境
        cleanup_test_environment() 
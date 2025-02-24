import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.system_validator import SystemValidator
from src.utils.logger import setup_logger
from src.model.llava_analyzer import LLaVAAnalyzer
from config.config import LLAVA_MODEL_PATH

logger = setup_logger('system_check')

def check_system():
    """检查系统环境和模型配置"""
    try:
        logger.info("开始系统检查...")
        
        # 1. 系统环境检查
        validator = SystemValidator()
        if not validator.validate_system():
            logger.error("系统环境检查未通过")
            return False
            
        # 获取系统信息
        system_info = validator.get_system_info()
        logger.info("\n=== 系统信息 ===")
        logger.info(f"操作系统: {system_info['platform']['system']}")
        logger.info(f"Python版本: {system_info['platform']['python_version']}")
        logger.info(f"PyTorch版本: {system_info['platform']['torch_version']}")
        logger.info(f"CPU核心数: {system_info['hardware']['cpu_count']}")
        logger.info(f"系统内存: {system_info['hardware']['memory_total']}")
        logger.info(f"磁盘空间: {system_info['hardware']['disk_total']}")
        
        if system_info['gpu']:
            logger.info("\nGPU信息:")
            for gpu in system_info['gpu']:
                logger.info(f"- {gpu['name']} ({gpu['memory']})")
        
        # 2. 资源监控
        resources = validator.monitor_resources()
        logger.info("\n=== 资源使用情况 ===")
        logger.info(f"CPU使用率: {resources['cpu']['usage']}%")
        logger.info(f"内存使用率: {resources['memory']['percent']}%")
        logger.info(f"磁盘使用率: {resources['disk']['percent']}%")
        
        if resources['gpu']:
            logger.info("\nGPU使用情况:")
            for gpu in resources['gpu']:
                logger.info(f"GPU {gpu['id']} ({gpu['name']}):")
                logger.info(f"- 显存使用: {gpu['allocated_memory']:.1f}GB / {gpu['total_memory']:.1f}GB")
                logger.info(f"- 利用率: {gpu['utilization']}%")
        
        # 3. 模型加载测试
        logger.info("\n=== 模型加载测试 ===")
        logger.info(f"LLaVA模型路径: {LLAVA_MODEL_PATH}")
        
        try:
            analyzer = LLaVAAnalyzer()
            logger.info("LLaVA模型加载成功")
        except Exception as e:
            logger.error(f"LLaVA模型加载失败: {e}")
            return False
        
        # 4. 清理资源
        validator.cleanup_resources()
        logger.info("\n系统检查完成!")
        return True
        
    except Exception as e:
        logger.error(f"系统检查过程出错: {e}")
        return False

if __name__ == "__main__":
    check_system() 
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

def test_logging():
    """测试日志记录功能"""
    try:
        # 创建不同类型的日志记录器
        monitor_logger = setup_logger('monitor')
        signal_logger = setup_logger('signal')
        debug_logger = setup_logger('debug')
        training_logger = setup_logger('training')
        
        print("开始测试日志记录功能...")
        
        # 监控日志测试
        monitor_logger.warning("监控警告：系统资源使用率较高")
        monitor_logger.error("监控错误：GPU显存不足")
        
        # 信号日志测试
        signal_logger.info("交易信号：发现潜在的SBS上升序列")
        signal_logger.info("交易信号：确认买入信号")
        
        # 调试日志测试
        debug_logger.debug("调试信息：正在加载模型")
        debug_logger.info("调试信息：模型加载完成")
        
        # 训练日志测试
        training_logger.info("训练进度：开始处理2024年1月数据")
        training_logger.warning("训练警告：发现异常序列")
        training_logger.error("训练错误：序列验证失败")
        
        print("日志记录测试完成，请检查logs目录下的日志文件")
        
    except Exception as e:
        print(f"测试过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_logging() 
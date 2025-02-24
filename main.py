import os
import signal
import sys
from dotenv import load_dotenv

from src.scheduler.task_scheduler import TaskScheduler
from src.utils.logger import setup_logger

# 加载环境变量
load_dotenv()

# 设置日志记录器
logger = setup_logger('main')

def signal_handler(signum, frame):
    """处理退出信号"""
    logger.info("Received exit signal. Cleaning up...")
    if scheduler:
        scheduler.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 检查必要的环境变量
    required_env_vars = ['DISCORD_WEBHOOK_URL', 'LLAVA_MODEL_PATH']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    try:
        # 创建并启动调度器
        scheduler = TaskScheduler()
        scheduler.start()
        
        logger.info("System started successfully. Press Ctrl+C to exit.")
        
        # 保持主线程运行
        signal.pause()
        
    except Exception as e:
        logger.error(f"System error: {e}")
        if 'scheduler' in locals():
            scheduler.cleanup()
        sys.exit(1) 
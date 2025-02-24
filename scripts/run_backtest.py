import os
import sys
from datetime import datetime, timedelta
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.backtester import Backtester
from src.utils.logger import setup_logger

logger = setup_logger('backtest_runner')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行SBS交易系统回测')
    
    # 添加日期参数
    parser.add_argument('--start_date', type=str, 
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                       help='回测结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 如果没有指定日期，使用上个月
    if not args.start_date or not args.end_date:
        today = datetime.now()
        first_day = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day = today.replace(day=1) - timedelta(days=1)
        
        args.start_date = first_day.strftime("%Y-%m-%d")
        args.end_date = last_day.strftime("%Y-%m-%d")
    
    return args

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")
        
        # 创建并运行回测器
        backtester = Backtester(args.start_date, args.end_date)
        backtester.run_backtest()
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
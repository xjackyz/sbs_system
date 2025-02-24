import pandas as pd
import os
from datetime import datetime, timedelta
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger('data_preparation')

def prepare_monthly_data(source_file: str, year: int = 2024):
    """
    按月处理原始数据并保存
    
    Args:
        source_file: 原始数据文件路径
        year: 目标年份
    """
    try:
        # 读取原始数据
        logger.info(f"正在读取数据: {source_file}")
        df = pd.read_csv(source_file)
        
        # 确保datetime列存在
        if 'datetime' not in df.columns:
            # 假设第一列是datetime
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 设置datetime为索引
        df.set_index('datetime', inplace=True)
        
        # 创建输出目录
        os.makedirs('data', exist_ok=True)
        
        # 按月处理数据
        for month in range(1, 13):
            # 提取当月数据
            mask = (df.index.year == year) & (df.index.month == month)
            monthly_data = df[mask].copy()
            
            if len(monthly_data) == 0:
                logger.warning(f"{year}年{month}月没有数据")
                continue
            
            # 确保数据列名符合要求
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            rename_dict = {}
            for col in monthly_data.columns:
                col_lower = col.lower()
                if col_lower in required_columns and col != col_lower:
                    rename_dict[col] = col_lower
            if rename_dict:
                monthly_data.rename(columns=rename_dict, inplace=True)
            
            # 保存月度数据
            output_file = f"data/NQ1!_{year}{month:02d}_1m.csv"
            monthly_data.to_csv(output_file)
            logger.info(f"保存{year}年{month}月数据: {output_file} ({len(monthly_data)}行)")
            
        logger.info("数据准备完成")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        sys.exit(1)

def prepare_date_range_data(source_file: str, start_date: str, end_date: str):
    """
    处理指定日期范围的数据
    
    Args:
        source_file: 原始数据文件路径
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """
    try:
        # 读取原始数据
        logger.info(f"正在读取数据: {source_file}")
        df = pd.read_csv(source_file)
        
        # 确保datetime列存在
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 设置datetime为索引
        df.set_index('datetime', inplace=True)
        
        # 过滤日期范围
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_data = df[mask].copy()
        
        if len(filtered_data) == 0:
            logger.warning(f"指定日期范围 {start_date} 到 {end_date} 没有数据")
            return None
        
        # 确保数据列名符合要求
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        rename_dict = {}
        for col in filtered_data.columns:
            col_lower = col.lower()
            if col_lower in required_columns and col != col_lower:
                rename_dict[col] = col_lower
        if rename_dict:
            filtered_data.rename(columns=rename_dict, inplace=True)
        
        # 创建输出目录
        os.makedirs('data', exist_ok=True)
        
        # 生成文件名
        start_str = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end_str = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        output_file = f"data/NQ1!_{start_str}_{end_str}_1m.csv"
        
        # 保存数据
        filtered_data.to_csv(output_file)
        logger.info(f"保存数据: {output_file} ({len(filtered_data)}行)")
        
        return output_file
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        return None

if __name__ == "__main__":
    source_file = '/home/easyai/桌面/nq/NQ_full_1min_continuous.csv'
    
    # 准备2024年全年数据
    prepare_monthly_data(source_file, 2024)
    
    # 或者准备指定日期范围的数据
    # prepare_date_range_data(source_file, "2024-01-01", "2024-12-31") 
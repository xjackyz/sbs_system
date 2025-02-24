"""数据服务层"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import logging
from sqlalchemy.orm import Session

from .database import DatabaseManager
from .repository import (
    MarketDataRepository,
    TechnicalIndicatorRepository,
    BacktestResultRepository,
    SignalRecordRepository
)
from .models import TimeFrame

class DataService:
    """数据服务"""
    
    def __init__(self, db_manager: DatabaseManager):
        """初始化数据服务
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def import_market_data(self,
                          data: pd.DataFrame,
                          symbol: str,
                          timeframe: TimeFrame) -> bool:
        """导入市场数据
        
        Args:
            data: 市场数据DataFrame
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            bool: 是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                repo = MarketDataRepository(session)
                
                # 准备数据
                data_list = []
                for index, row in data.iterrows():
                    data_list.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': index,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row.get('volume', 0)
                    })
                
                # 批量导入
                return repo.batch_add_market_data(data_list)
                
        except Exception as e:
            self.logger.error(f"导入市场数据失败: {str(e)}")
            return False
            
    def get_market_data(self,
                       symbol: str,
                       timeframe: TimeFrame,
                       start_time: datetime,
                       end_time: datetime) -> pd.DataFrame:
        """获取市场数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            pd.DataFrame: 市场数据
        """
        try:
            with self.db_manager.get_session() as session:
                repo = MarketDataRepository(session)
                return repo.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return pd.DataFrame()
            
    def save_technical_indicator(self,
                               market_data_id: int,
                               name: str,
                               parameters: Dict[str, Any],
                               values: Dict[str, Any]) -> bool:
        """保存技术指标
        
        Args:
            market_data_id: 市场数据ID
            name: 指标名称
            parameters: 指标参数
            values: 指标值
            
        Returns:
            bool: 是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                repo = TechnicalIndicatorRepository(session)
                indicator = repo.add_indicator({
                    'market_data_id': market_data_id,
                    'name': name,
                    'parameters': parameters,
                    'values': values
                })
                return indicator is not None
        except Exception as e:
            self.logger.error(f"保存技术指标失败: {str(e)}")
            return False
            
    def save_backtest_result(self, data: Dict[str, Any]) -> bool:
        """保存回测结果
        
        Args:
            data: 回测结果数据
            
        Returns:
            bool: 是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                repo = BacktestResultRepository(session)
                result = repo.add_result(data)
                return result is not None
        except Exception as e:
            self.logger.error(f"保存回测结果失败: {str(e)}")
            return False
            
    def get_backtest_results(self,
                           strategy_name: Optional[str] = None,
                           symbol: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取回测结果
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict]: 回测结果列表
        """
        try:
            with self.db_manager.get_session() as session:
                repo = BacktestResultRepository(session)
                results = repo.get_results(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time
                )
                return [self._convert_to_dict(result) for result in results]
        except Exception as e:
            self.logger.error(f"获取回测结果失败: {str(e)}")
            return []
            
    def save_signal(self, data: Dict[str, Any]) -> Optional[int]:
        """保存交易信号
        
        Args:
            data: 信号数据
            
        Returns:
            Optional[int]: 信号ID
        """
        try:
            with self.db_manager.get_session() as session:
                repo = SignalRecordRepository(session)
                signal = repo.add_signal(data)
                return signal.id if signal else None
        except Exception as e:
            self.logger.error(f"保存交易信号失败: {str(e)}")
            return None
            
    def update_signal(self,
                     signal_id: int,
                     status: str,
                     result: Optional[str] = None,
                     profit_loss: Optional[float] = None) -> bool:
        """更新信号状态
        
        Args:
            signal_id: 信号ID
            status: 新状态
            result: 交易结果
            profit_loss: 盈亏
            
        Returns:
            bool: 是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                repo = SignalRecordRepository(session)
                signal = repo.update_signal_status(
                    id=signal_id,
                    status=status,
                    result=result,
                    profit_loss=profit_loss
                )
                return signal is not None
        except Exception as e:
            self.logger.error(f"更新信号状态失败: {str(e)}")
            return False
            
    def get_signals(self,
                   symbol: Optional[str] = None,
                   timeframe: Optional[TimeFrame] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取交易信号
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            status: 信号状态
            
        Returns:
            List[Dict]: 信号列表
        """
        try:
            with self.db_manager.get_session() as session:
                repo = SignalRecordRepository(session)
                signals = repo.get_signals(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    status=status
                )
                return [self._convert_to_dict(signal) for signal in signals]
        except Exception as e:
            self.logger.error(f"获取交易信号失败: {str(e)}")
            return []
            
    @staticmethod
    def _convert_to_dict(obj) -> Dict[str, Any]:
        """转换对象为字典"""
        if not hasattr(obj, '__dict__'):
            return {}
            
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, TimeFrame):
                    result[key] = value.value
                else:
                    result[key] = value
        return result 
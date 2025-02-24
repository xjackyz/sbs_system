"""数据访问层"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

from .models import MarketData, TechnicalIndicator, BacktestResult, SignalRecord, TimeFrame

class MarketDataRepository:
    """市场数据仓库"""
    
    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)
        
    def add_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """添加市场数据"""
        try:
            market_data = MarketData(**data)
            self.session.add(market_data)
            self.session.commit()
            self.session.refresh(market_data)
            return market_data
        except Exception as e:
            self.logger.error(f"添加市场数据失败: {str(e)}")
            self.session.rollback()
            return None
            
    def batch_add_market_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量添加市场数据"""
        try:
            market_data_objects = [MarketData(**data) for data in data_list]
            self.session.bulk_save_objects(market_data_objects)
            self.session.commit()
            return True
        except Exception as e:
            self.logger.error(f"批量添加市场数据失败: {str(e)}")
            self.session.rollback()
            return False
            
    def get_market_data(self,
                       symbol: str,
                       timeframe: TimeFrame,
                       start_time: datetime,
                       end_time: datetime) -> pd.DataFrame:
        """获取市场数据"""
        try:
            query = self.session.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.timestamp >= start_time,
                    MarketData.timestamp <= end_time
                )
            ).order_by(MarketData.timestamp)
            
            # 转换为DataFrame
            records = query.all()
            if not records:
                return pd.DataFrame()
                
            data = []
            for record in records:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return pd.DataFrame()
            
    def update_market_data(self,
                          id: int,
                          data: Dict[str, Any]) -> Optional[MarketData]:
        """更新市场数据"""
        try:
            market_data = self.session.query(MarketData).filter(
                MarketData.id == id
            ).first()
            
            if not market_data:
                return None
                
            for key, value in data.items():
                setattr(market_data, key, value)
                
            self.session.commit()
            self.session.refresh(market_data)
            return market_data
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {str(e)}")
            self.session.rollback()
            return None

class TechnicalIndicatorRepository:
    """技术指标仓库"""
    
    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)
        
    def add_indicator(self, data: Dict[str, Any]) -> Optional[TechnicalIndicator]:
        """添加技术指标"""
        try:
            indicator = TechnicalIndicator(**data)
            self.session.add(indicator)
            self.session.commit()
            self.session.refresh(indicator)
            return indicator
        except Exception as e:
            self.logger.error(f"添加技术指标失败: {str(e)}")
            self.session.rollback()
            return None
            
    def get_indicators(self,
                      market_data_id: int,
                      indicator_name: Optional[str] = None) -> List[TechnicalIndicator]:
        """获取技术指标"""
        try:
            query = self.session.query(TechnicalIndicator).filter(
                TechnicalIndicator.market_data_id == market_data_id
            )
            
            if indicator_name:
                query = query.filter(TechnicalIndicator.name == indicator_name)
                
            return query.all()
            
        except Exception as e:
            self.logger.error(f"获取技术指标失败: {str(e)}")
            return []

class BacktestResultRepository:
    """回测结果仓库"""
    
    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)
        
    def add_result(self, data: Dict[str, Any]) -> Optional[BacktestResult]:
        """添加回测结果"""
        try:
            result = BacktestResult(**data)
            self.session.add(result)
            self.session.commit()
            self.session.refresh(result)
            return result
        except Exception as e:
            self.logger.error(f"添加回测结果失败: {str(e)}")
            self.session.rollback()
            return None
            
    def get_results(self,
                   strategy_name: Optional[str] = None,
                   symbol: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[BacktestResult]:
        """获取回测结果"""
        try:
            query = self.session.query(BacktestResult)
            
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if start_time:
                query = query.filter(BacktestResult.start_time >= start_time)
            if end_time:
                query = query.filter(BacktestResult.end_time <= end_time)
                
            return query.order_by(desc(BacktestResult.created_at)).all()
            
        except Exception as e:
            self.logger.error(f"获取回测结果失败: {str(e)}")
            return []

class SignalRecordRepository:
    """交易信号记录仓库"""
    
    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)
        
    def add_signal(self, data: Dict[str, Any]) -> Optional[SignalRecord]:
        """添加交易信号"""
        try:
            signal = SignalRecord(**data)
            self.session.add(signal)
            self.session.commit()
            self.session.refresh(signal)
            return signal
        except Exception as e:
            self.logger.error(f"添加交易信号失败: {str(e)}")
            self.session.rollback()
            return None
            
    def update_signal_status(self,
                           id: int,
                           status: str,
                           result: Optional[str] = None,
                           profit_loss: Optional[float] = None) -> Optional[SignalRecord]:
        """更新信号状态"""
        try:
            signal = self.session.query(SignalRecord).filter(
                SignalRecord.id == id
            ).first()
            
            if not signal:
                return None
                
            signal.status = status
            if result:
                signal.result = result
            if profit_loss is not None:
                signal.profit_loss = profit_loss
                
            self.session.commit()
            self.session.refresh(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"更新信号状态失败: {str(e)}")
            self.session.rollback()
            return None
            
    def get_signals(self,
                   symbol: Optional[str] = None,
                   timeframe: Optional[TimeFrame] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   status: Optional[str] = None) -> List[SignalRecord]:
        """获取交易信号"""
        try:
            query = self.session.query(SignalRecord)
            
            if symbol:
                query = query.filter(SignalRecord.symbol == symbol)
            if timeframe:
                query = query.filter(SignalRecord.timeframe == timeframe)
            if start_time:
                query = query.filter(SignalRecord.timestamp >= start_time)
            if end_time:
                query = query.filter(SignalRecord.timestamp <= end_time)
            if status:
                query = query.filter(SignalRecord.status == status)
                
            return query.order_by(desc(SignalRecord.timestamp)).all()
            
        except Exception as e:
            self.logger.error(f"获取交易信号失败: {str(e)}")
            return [] 
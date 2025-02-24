"""数据模型定义"""
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .database import Base

class TimeFrame(enum.Enum):
    """时间周期"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class MarketData(Base):
    """市场数据"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(Enum(TimeFrame), index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联的技术指标
    indicators = relationship("TechnicalIndicator", back_populates="market_data")
    
    class Config:
        orm_mode = True

class TechnicalIndicator(Base):
    """技术指标"""
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))
    name = Column(String, index=True)  # 指标名称
    parameters = Column(JSON)  # 指标参数
    values = Column(JSON)  # 指标值
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联的市场数据
    market_data = relationship("MarketData", back_populates="indicators")
    
    class Config:
        orm_mode = True

class BacktestResult(Base):
    """回测结果"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(Enum(TimeFrame))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    parameters = Column(JSON)  # 策略参数
    trades = Column(JSON)  # 交易记录
    equity_curve = Column(JSON)  # 权益曲线
    created_at = Column(DateTime, default=datetime.utcnow)
    
    class Config:
        orm_mode = True

class SignalRecord(Base):
    """交易信号记录"""
    __tablename__ = "signal_records"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(Enum(TimeFrame))
    timestamp = Column(DateTime, index=True)
    signal_type = Column(String)  # LONG/SHORT
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Float)
    status = Column(String)  # PENDING/TRIGGERED/COMPLETED/CANCELLED
    result = Column(String)  # WIN/LOSS/NONE
    profit_loss = Column(Float)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class Config:
        orm_mode = True 
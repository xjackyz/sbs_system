"""
投资组合管理模块，用于管理回测过程中的持仓和资金
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from src.utils.logger import setup_logger

logger = setup_logger('portfolio')

@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    direction: str  # 'long' 或 'short'
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    exit_time: datetime = None
    exit_price: float = None
    profit_loss: float = 0.0
    commission: float = 0.0
    status: str = 'open'  # 'open' 或 'closed'
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.entry_time is None:
            self.entry_time = datetime.now()
            
    def close(self, exit_price: float, exit_time: datetime = None):
        """关闭持仓"""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.status = 'closed'
        
        # 计算盈亏
        if self.direction == 'long':
            self.profit_loss = (self.exit_price - self.entry_price) * self.size
        else:
            self.profit_loss = (self.entry_price - self.exit_price) * self.size
            
        # 扣除手续费
        self.profit_loss -= self.commission
        
    def update(self, current_price: float) -> bool:
        """
        更新持仓状态
        
        Args:
            current_price: 当前价格
            
        Returns:
            bool: 是否需要平仓
        """
        if self.status == 'closed':
            return False
            
        # 检查止损
        if self.stop_loss is not None:
            if (self.direction == 'long' and current_price <= self.stop_loss) or \
               (self.direction == 'short' and current_price >= self.stop_loss):
                return True
                
        # 检查止盈
        if self.take_profit is not None:
            if (self.direction == 'long' and current_price >= self.take_profit) or \
               (self.direction == 'short' and current_price <= self.take_profit):
                return True
                
        return False

class Portfolio:
    """投资组合类"""
    def __init__(self, initial_capital: float):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.cash = initial_capital
        self.equity = initial_capital
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        
    def open_position(self, signal: Dict) -> Optional[Position]:
        """
        开仓
        
        Args:
            signal: 交易信号
            
        Returns:
            Position: 新建仓位或None
        """
        try:
            # 检查资金是否足够
            required_capital = signal['size'] * signal['entry_price']
            if required_capital > self.cash:
                logger.warning(f"资金不足: 需要 {required_capital}, 可用 {self.cash}")
                return None
                
            # 创建新仓位
            position = Position(
                symbol=signal['symbol'],
                direction=signal['direction'],
                entry_price=signal['entry_price'],
                size=signal['size'],
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                commission=signal.get('commission', 0.0)
            )
            
            # 更新资金
            self.cash -= required_capital + position.commission
            self.positions.append(position)
            
            logger.info(f"开仓成功: {position}")
            return position
            
        except Exception as e:
            logger.error(f"开仓失败: {e}")
            return None
            
    def close_position(self, position: Position, exit_price: float,
                      exit_time: datetime = None) -> bool:
        """
        平仓
        
        Args:
            position: 要平仓的持仓
            exit_price: 平仓价格
            exit_time: 平仓时间
            
        Returns:
            bool: 是否成功
        """
        try:
            if position not in self.positions:
                logger.warning("找不到要平仓的持仓")
                return False
                
            # 平仓
            position.close(exit_price, exit_time)
            
            # 更新资金
            self.cash += position.size * exit_price - position.commission
            self.positions.remove(position)
            self.closed_positions.append(position)
            
            # 更新权益
            self._update_equity()
            
            logger.info(f"平仓成功: {position}")
            return True
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return False
            
    def update_positions(self, current_prices: Dict[str, float]):
        """
        更新所有持仓状态
        
        Args:
            current_prices: 当前价格字典 {symbol: price}
        """
        for position in self.positions[:]:  # 创建副本以避免在迭代时修改
            current_price = current_prices.get(position.symbol)
            if current_price is None:
                continue
                
            if position.update(current_price):
                self.close_position(position, current_price)
                
        self._update_equity()
        
    def _update_equity(self):
        """更新权益"""
        self.equity = self.cash + sum(
            pos.size * pos.entry_price for pos in self.positions
        )
        
        # 更新最大权益和最大回撤
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
    def get_statistics(self) -> Dict:
        """获取投资组合统计数据"""
        stats = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'cash': self.cash,
            'equity': self.equity,
            'max_equity': self.max_equity,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_pnl': sum(pos.profit_loss for pos in self.closed_positions),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        return stats
        
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.closed_positions:
            return 0.0
            
        winning_trades = sum(1 for pos in self.closed_positions if pos.profit_loss > 0)
        return winning_trades / len(self.closed_positions)
        
    def _calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        gross_profit = sum(pos.profit_loss for pos in self.closed_positions if pos.profit_loss > 0)
        gross_loss = abs(sum(pos.profit_loss for pos in self.closed_positions if pos.profit_loss < 0))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf') 
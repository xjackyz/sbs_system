"""评估器模块"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from .metrics import TradingMetrics, ModelMetrics, ValidationMetrics
from ...data_management.service import DataService

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, data_service: DataService, config: Dict[str, Any]):
        """初始化评估器
        
        Args:
            data_service: 数据服务实例
            config: 评估配置
        """
        self.data_service = data_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        
    def evaluate_model(self,
                      model: Any,
                      train_data: Tuple[np.ndarray, np.ndarray],
                      val_data: Tuple[np.ndarray, np.ndarray],
                      test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """评估模型性能
        
        Args:
            model: 模型实例
            train_data: 训练数据元组 (X_train, y_train)
            val_data: 验证数据元组 (X_val, y_val)
            test_data: 测试数据元组 (X_test, y_test)
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 获取预测结果
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # 计算训练集和验证集指标
            train_metrics = ModelMetrics.calculate_classification_metrics(y_train, train_pred)
            val_metrics = ModelMetrics.calculate_classification_metrics(y_val, val_pred)
            
            # 计算过拟合指标
            overfitting_metrics = ValidationMetrics.calculate_overfitting_metrics(
                train_metrics, val_metrics
            )
            
            # 计算模型稳定性指标
            stability_metrics = self._calculate_model_stability(model, X_val)
            
            # 如果有测试集，计算测试集指标
            test_metrics = {}
            if test_data is not None:
                X_test, y_test = test_data
                test_pred = model.predict(X_test)
                test_metrics = ModelMetrics.calculate_classification_metrics(y_test, test_pred)
            
            # 合并所有指标
            evaluation_results = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'overfitting_metrics': overfitting_metrics,
                'stability_metrics': stability_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # 保存评估历史
            self.metrics_history.append(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}")
            return {}
            
    def evaluate_trading_strategy(self,
                                trades: List[Dict[str, Any]],
                                equity_curve: np.ndarray,
                                returns: np.ndarray) -> Dict[str, Any]:
        """评估交易策略性能
        
        Args:
            trades: 交易记录列表
            equity_curve: 权益曲线
            returns: 收益率序列
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 计算交易相关指标
            trading_metrics = {
                'sharpe_ratio': TradingMetrics.calculate_sharpe_ratio(returns),
                'max_drawdown': TradingMetrics.calculate_max_drawdown(equity_curve),
                'win_rate': TradingMetrics.calculate_win_rate(trades),
                'profit_factor': TradingMetrics.calculate_profit_factor(trades)
            }
            
            # 计算其他统计指标
            total_trades = len(trades)
            total_profit = sum(trade['profit_loss'] for trade in trades)
            avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
            
            trading_stats = {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'avg_profit_per_trade': avg_profit_per_trade,
                'final_equity': float(equity_curve[-1]) if len(equity_curve) > 0 else 0
            }
            
            # 保存回测结果
            backtest_result = {
                'strategy_name': self.config.get('strategy_name', 'unknown'),
                'symbol': self.config.get('symbol', 'unknown'),
                'timeframe': self.config.get('timeframe', 'unknown'),
                'start_time': trades[0]['timestamp'] if trades else None,
                'end_time': trades[-1]['timestamp'] if trades else None,
                'initial_capital': float(equity_curve[0]) if len(equity_curve) > 0 else 0,
                'final_capital': float(equity_curve[-1]) if len(equity_curve) > 0 else 0,
                'total_trades': total_trades,
                'winning_trades': sum(1 for trade in trades if trade['profit_loss'] > 0),
                'losing_trades': sum(1 for trade in trades if trade['profit_loss'] < 0),
                'metrics': trading_metrics,
                'equity_curve': equity_curve.tolist(),
                'trades': trades
            }
            
            # 保存到数据库
            self.data_service.save_backtest_result(backtest_result)
            
            return {
                'trading_metrics': trading_metrics,
                'trading_stats': trading_stats,
                'backtest_result': backtest_result
            }
            
        except Exception as e:
            self.logger.error(f"交易策略评估失败: {str(e)}")
            return {}
            
    def _calculate_model_stability(self, model: Any, X: np.ndarray, n_runs: int = 5) -> Dict[str, float]:
        """计算模型稳定性
        
        Args:
            model: 模型实例
            X: 输入数据
            n_runs: 运行次数
            
        Returns:
            Dict[str, float]: 稳定性指标
        """
        predictions = []
        
        # 使用线程池并行计算多次预测
        with ThreadPoolExecutor(max_workers=min(n_runs, 5)) as executor:
            futures = [executor.submit(model.predict, X) for _ in range(n_runs)]
            predictions = [future.result() for future in futures]
            
        return ValidationMetrics.calculate_stability_metrics(predictions)
        
    def save_metrics_history(self, filepath: str):
        """保存评估历史
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存评估历史失败: {str(e)}")
            
    def load_metrics_history(self, filepath: str):
        """加载评估历史
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r') as f:
                self.metrics_history = json.load(f)
        except Exception as e:
            self.logger.error(f"加载评估历史失败: {str(e)}")
            self.metrics_history = [] 
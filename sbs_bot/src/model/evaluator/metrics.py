"""评估指标模块"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class TradingMetrics:
    """交易相关的评估指标"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            float: 夏普比率
        """
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)
        
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """计算最大回撤
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            float: 最大回撤
        """
        if len(equity_curve) < 2:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        return np.max(drawdowns)
        
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """计算胜率
        
        Args:
            trades: 交易记录列表
            
        Returns:
            float: 胜率
        """
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
        return winning_trades / len(trades)
        
    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """计算盈亏比
        
        Args:
            trades: 交易记录列表
            
        Returns:
            float: 盈亏比
        """
        if not trades:
            return 0.0
        gross_profit = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0)
        gross_loss = abs(sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0.0

class ModelMetrics:
    """模型相关的评估指标"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray,
                                      y_pred: np.ndarray) -> Dict[str, float]:
        """计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            Dict[str, float]: 指标字典
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict[str, float]: 指标字典
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

class ValidationMetrics:
    """验证相关的评估指标"""
    
    @staticmethod
    def calculate_overfitting_metrics(train_metrics: Dict[str, float],
                                    val_metrics: Dict[str, float]) -> Dict[str, float]:
        """计算过拟合指标
        
        Args:
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            
        Returns:
            Dict[str, float]: 过拟合指标
        """
        metrics_diff = {}
        for metric in train_metrics:
            if metric in val_metrics:
                diff = train_metrics[metric] - val_metrics[metric]
                metrics_diff[f'{metric}_diff'] = diff
                metrics_diff[f'{metric}_ratio'] = (
                    train_metrics[metric] / val_metrics[metric]
                    if val_metrics[metric] != 0 else float('inf')
                )
        return metrics_diff
        
    @staticmethod
    def calculate_stability_metrics(predictions: List[np.ndarray]) -> Dict[str, float]:
        """计算模型稳定性指标
        
        Args:
            predictions: 多次预测结果列表
            
        Returns:
            Dict[str, float]: 稳定性指标
        """
        if not predictions or len(predictions) < 2:
            return {'prediction_std': 0.0, 'prediction_cv': 0.0}
            
        predictions_array = np.array(predictions)
        prediction_std = np.std(predictions_array, axis=0)
        prediction_mean = np.mean(predictions_array, axis=0)
        prediction_cv = np.mean(prediction_std / prediction_mean)
        
        return {
            'prediction_std': float(np.mean(prediction_std)),
            'prediction_cv': float(prediction_cv)
        } 
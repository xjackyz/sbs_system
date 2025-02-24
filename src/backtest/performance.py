"""
性能分析模块
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

from src.utils.logger import setup_logger

logger = setup_logger('performance_analyzer')

class PerformanceAnalyzer:
    """性能分析器"""
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化性能分析器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        计算性能指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict: 性能指标字典
        """
        try:
            # 基本统计量
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            daily_std = returns.std()
            annual_std = daily_std * np.sqrt(252)
            
            # 风险调整收益
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            
            # 回撤分析
            drawdown = self.calculate_drawdown(returns)
            max_drawdown = abs(drawdown.min())
            
            # 交易统计
            positive_days = len(returns[returns > 0])
            negative_days = len(returns[returns < 0])
            win_rate = positive_days / len(returns) if len(returns) > 0 else 0
            
            # 风险指标
            var_95 = self.calculate_var(returns, 0.95)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            
            # 收益分布
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_std,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'positive_days': positive_days,
                'negative_days': negative_days
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return {}
            
    def calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """
        计算回撤序列
        
        Args:
            returns: 收益率序列
            
        Returns:
            pd.Series: 回撤序列
        """
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = cumulative / rolling_max - 1
            return drawdown
            
        except Exception as e:
            logger.error(f"计算回撤失败: {e}")
            return pd.Series()
            
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 夏普比率
        """
        try:
            excess_returns = returns - self.risk_free_rate / 252
            if len(excess_returns) < 2:
                return 0.0
                
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
            
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 索提诺比率
        """
        try:
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) < 2:
                return 0.0
                
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            return np.sqrt(252) * excess_returns.mean() / downside_std
            
        except Exception as e:
            logger.error(f"计算索提诺比率失败: {e}")
            return 0.0
            
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            
        Returns:
            float: VaR值
        """
        try:
            return abs(np.percentile(returns, (1 - confidence) * 100))
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0
            
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算条件风险价值(CVaR)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            
        Returns:
            float: CVaR值
        """
        try:
            var = self.calculate_var(returns, confidence)
            return abs(returns[returns <= -var].mean())
            
        except Exception as e:
            logger.error(f"计算CVaR失败: {e}")
            return 0.0
            
    def calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """
        计算交易统计数据
        
        Args:
            trades: 交易记录列表
            
        Returns:
            Dict: 交易统计数据
        """
        try:
            if not trades:
                return {}
                
            profits = [t['profit_loss'] for t in trades]
            durations = [t['duration'].total_seconds() / 3600 for t in trades]  # 转换为小时
            
            stats = {
                'total_trades': len(trades),
                'profitable_trades': len([p for p in profits if p > 0]),
                'loss_trades': len([p for p in profits if p <= 0]),
                'total_profit': sum([p for p in profits if p > 0]),
                'total_loss': sum([p for p in profits if p <= 0]),
                'largest_profit': max(profits),
                'largest_loss': min(profits),
                'average_profit': np.mean([p for p in profits if p > 0]),
                'average_loss': np.mean([p for p in profits if p <= 0]),
                'win_rate': len([p for p in profits if p > 0]) / len(trades),
                'average_duration': np.mean(durations),
                'min_duration': min(durations),
                'max_duration': max(durations)
            }
            
            # 计算盈亏比
            if stats['average_loss'] != 0:
                stats['profit_loss_ratio'] = abs(stats['average_profit'] / stats['average_loss'])
            else:
                stats['profit_loss_ratio'] = float('inf')
                
            # 计算期望值
            stats['expectancy'] = (stats['average_profit'] * stats['win_rate'] +
                                 stats['average_loss'] * (1 - stats['win_rate']))
                                 
            return stats
            
        except Exception as e:
            logger.error(f"计算交易统计失败: {e}")
            return {}
            
    def plot_equity_curve(self, equity_curve: pd.Series, save_path: Optional[str] = None):
        """
        绘制权益曲线
        
        Args:
            equity_curve: 权益曲线数据
            save_path: 保存路径
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve.values, label='Equity')
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制权益曲线失败: {e}")
            
    def plot_drawdown(self, returns: pd.Series, save_path: Optional[str] = None):
        """
        绘制回撤图
        
        Args:
            returns: 收益率序列
            save_path: 保存路径
        """
        try:
            drawdown = self.calculate_drawdown(returns)
            
            plt.figure(figsize=(12, 6))
            plt.plot(drawdown.index, drawdown.values * 100, label='Drawdown')
            plt.title('Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制回撤图失败: {e}")
            
    def plot_monthly_returns(self, returns: pd.Series, save_path: Optional[str] = None):
        """
        绘制月度收益热图
        
        Args:
            returns: 收益率序列
            save_path: 保存路径
        """
        try:
            monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns.to_frame()
            monthly_returns.columns = ['Returns']
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(monthly_returns.pivot_table(
                values='Returns',
                index=monthly_returns.index.year,
                columns=monthly_returns.index.month,
                aggfunc='first'
            ) * 100,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn')
            
            plt.title('Monthly Returns (%)')
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制月度收益热图失败: {e}")
            
    def plot_distribution(self, returns: pd.Series, save_path: Optional[str] = None):
        """
        绘制收益分布图
        
        Args:
            returns: 收益率序列
            save_path: 保存路径
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制直方图
            sns.histplot(returns * 100, bins=50, kde=True)
            
            # 添加统计信息
            mean = returns.mean() * 100
            std = returns.std() * 100
            skew = returns.skew()
            kurt = returns.kurtosis()
            
            plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}%')
            plt.title(f'Returns Distribution\nSkew: {skew:.2f}, Kurt: {kurt:.2f}')
            plt.xlabel('Returns (%)')
            plt.ylabel('Frequency')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制收益分布图失败: {e}")
            
    def generate_report(self, returns: pd.Series, trades: List[Dict],
                       output_dir: str) -> Dict:
        """
        生成完整的性能报告
        
        Args:
            returns: 收益率序列
            trades: 交易记录列表
            output_dir: 输出目录
            
        Returns:
            Dict: 报告数据
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 计算各项指标
            metrics = self.calculate_metrics(returns)
            trade_stats = self.calculate_trade_statistics(trades)
            
            # 生成图表
            self.plot_equity_curve(
                (1 + returns).cumprod(),
                os.path.join(output_dir, 'equity_curve.png')
            )
            self.plot_drawdown(
                returns,
                os.path.join(output_dir, 'drawdown.png')
            )
            self.plot_monthly_returns(
                returns,
                os.path.join(output_dir, 'monthly_returns.png')
            )
            self.plot_distribution(
                returns,
                os.path.join(output_dir, 'distribution.png')
            )
            
            # 合并报告数据
            report = {
                'performance_metrics': metrics,
                'trade_statistics': trade_stats,
                'report_time': datetime.now().isoformat(),
                'charts': {
                    'equity_curve': 'equity_curve.png',
                    'drawdown': 'drawdown.png',
                    'monthly_returns': 'monthly_returns.png',
                    'distribution': 'distribution.png'
                }
            }
            
            # 保存报告数据
            pd.to_json(
                os.path.join(output_dir, 'report.json'),
                report,
                indent=4,
                orient='index'
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {} 
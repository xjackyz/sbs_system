import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.backtester import Backtester, BacktestConfig
from src.backtest.portfolio import Portfolio
from src.backtest.performance import PerformanceAnalyzer

class TestBacktester(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=100000,
            position_size=0.1,
            max_positions=5,
            commission_rate=0.001,
            slippage=0.0005,
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1h', '4h', '1d'],
            risk_free_rate=0.02
        )
        
        # 创建测试数据
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='1H')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(40000, 1000, len(dates)),
            'high': np.random.normal(41000, 1000, len(dates)),
            'low': np.random.normal(39000, 1000, len(dates)),
            'close': np.random.normal(40500, 1000, len(dates)),
            'volume': np.random.normal(1000, 100, len(dates))
        })
        
        self.backtester = Backtester(self.config)
        self.portfolio = Portfolio(self.config.initial_capital)
        self.analyzer = PerformanceAnalyzer()
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.backtester.config.initial_capital, 100000)
        self.assertEqual(self.backtester.config.position_size, 0.1)
        self.assertEqual(len(self.backtester.config.symbols), 2)
        
    def test_load_data(self):
        """测试数据加载"""
        data = self.backtester.load_data('BTCUSDT', '1h')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue('open' in data.columns)
        self.assertTrue('close' in data.columns)
        
    def test_preprocess_data(self):
        """测试数据预处理"""
        processed_data = self.backtester.preprocess_data(self.test_data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertTrue('returns' in processed_data.columns)
        self.assertTrue('volatility' in processed_data.columns)
        
    def test_generate_signals(self):
        """测试信号生成"""
        signals = self.backtester.generate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.Series)
        self.assertTrue(all(s in [-1, 0, 1] for s in signals.unique()))
        
    def test_execute_trades(self):
        """测试交易执行"""
        signals = pd.Series(np.random.choice([-1, 0, 1], len(self.test_data)))
        trades = self.backtester.execute_trades(self.test_data, signals)
        
        self.assertIsInstance(trades, list)
        for trade in trades:
            self.assertTrue(hasattr(trade, 'entry_price'))
            self.assertTrue(hasattr(trade, 'exit_price'))
            self.assertTrue(hasattr(trade, 'profit_loss'))
            
    def test_calculate_position_size(self):
        """测试仓位大小计算"""
        position_size = self.backtester.calculate_position_size(
            capital=100000,
            price=40000,
            risk_per_trade=0.01
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
        
    def test_apply_slippage(self):
        """测试滑点应用"""
        price = 40000
        slipped_price = self.backtester.apply_slippage(price, 'buy')
        
        self.assertGreater(slipped_price, price)
        
        slipped_price = self.backtester.apply_slippage(price, 'sell')
        self.assertLess(slipped_price, price)
        
    def test_calculate_commission(self):
        """测试手续费计算"""
        trade_value = 40000
        commission = self.backtester.calculate_commission(trade_value)
        
        expected_commission = trade_value * self.config.commission_rate
        self.assertEqual(commission, expected_commission)
        
    def test_portfolio_update(self):
        """测试投资组合更新"""
        self.portfolio.update(
            trade_type='buy',
            symbol='BTCUSDT',
            quantity=1,
            price=40000,
            commission=40
        )
        
        self.assertEqual(len(self.portfolio.positions), 1)
        self.assertLess(self.portfolio.cash, self.config.initial_capital)
        
    def test_risk_management(self):
        """测试风险管理"""
        position = {
            'symbol': 'BTCUSDT',
            'quantity': 1,
            'entry_price': 40000
        }
        
        stop_loss = self.backtester.calculate_stop_loss(position, 0.02)
        self.assertLess(stop_loss, position['entry_price'])
        
        take_profit = self.backtester.calculate_take_profit(position, 0.04)
        self.assertGreater(take_profit, position['entry_price'])
        
    def test_performance_metrics(self):
        """测试性能指标计算"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = self.analyzer.calculate_metrics(returns)
        
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        equity_curve = pd.Series([100000, 102000, 101000, 103000, 99000])
        drawdown = self.analyzer.calculate_drawdown(equity_curve)
        
        self.assertIsInstance(drawdown, pd.Series)
        self.assertTrue(all(d <= 0 for d in drawdown))
        
    def test_trade_statistics(self):
        """测试交易统计"""
        trades = [
            {'profit_loss': 1000, 'duration': timedelta(hours=2)},
            {'profit_loss': -500, 'duration': timedelta(hours=1)},
            {'profit_loss': 1500, 'duration': timedelta(hours=3)}
        ]
        
        stats = self.analyzer.calculate_trade_statistics(trades)
        
        self.assertIn('total_trades', stats)
        self.assertIn('profitable_trades', stats)
        self.assertIn('average_profit', stats)
        self.assertIn('average_duration', stats)
        
    def test_optimization(self):
        """测试参数优化"""
        param_grid = {
            'position_size': [0.1, 0.2],
            'stop_loss': [0.02, 0.03]
        }
        
        results = self.backtester.optimize_parameters(
            self.test_data,
            param_grid,
            metric='sharpe_ratio'
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('best_params', results)
        self.assertIn('best_score', results)
        
    def test_cross_validation(self):
        """测试交叉验证"""
        cv_results = self.backtester.cross_validate(
            self.test_data,
            n_splits=5,
            metric='sharpe_ratio'
        )
        
        self.assertIsInstance(cv_results, dict)
        self.assertEqual(len(cv_results['scores']), 5)
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        
    def test_report_generation(self):
        """测试报告生成"""
        trades = [
            {'profit_loss': 1000, 'duration': timedelta(hours=2)},
            {'profit_loss': -500, 'duration': timedelta(hours=1)}
        ]
        equity_curve = pd.Series([100000, 101000, 100500])
        
        report = self.backtester.generate_report(trades, equity_curve)
        
        self.assertIn('performance_metrics', report)
        self.assertIn('trade_statistics', report)
        self.assertIn('equity_curve', report)
        
    def tearDown(self):
        """测试后的清理"""
        self.backtester = None
        self.portfolio = None
        self.analyzer = None

if __name__ == '__main__':
    unittest.main() 
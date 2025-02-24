import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger('visualization')

class VisualizationSystem:
    """可视化系统"""
    
    def __init__(self, save_dir: str = 'visualization'):
        """
        初始化可视化系统
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_real_time_chart(self, 
                             data: pd.DataFrame,
                             patterns: List[Dict],
                             signals: List[Dict]) -> go.Figure:
        """
        创建实时图表
        
        Args:
            data: K线数据
            patterns: SBS模式列表
            signals: 交易信号列表
            
        Returns:
            Plotly图表对象
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )
            
            # 添加K线图
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='K线'
                ),
                row=1, col=1
            )
            
            # 添加SBS模式标记
            for pattern in patterns:
                fig.add_trace(
                    go.Scatter(
                        x=[pattern['start_time'], pattern['end_time']],
                        y=[pattern['low_price'], pattern['high_price']],
                        mode='lines+markers',
                        name=f'SBS模式 ({pattern["confidence"]:.2f})',
                        line=dict(color='rgba(255, 165, 0, 0.5)', width=2)
                    ),
                    row=1, col=1
                )
            
            # 添加交易信号
            for signal in signals:
                fig.add_trace(
                    go.Scatter(
                        x=[signal['time']],
                        y=[signal['price']],
                        mode='markers',
                        name=f'{signal["type"]} ({signal["confidence"]:.2f})',
                        marker=dict(
                            symbol='triangle-up' if signal['type'] == 'buy' else 'triangle-down',
                            size=15,
                            color='green' if signal['type'] == 'buy' else 'red'
                        )
                    ),
                    row=1, col=1
                )
            
            # 添加性能指标
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['performance'],
                    name='性能指标',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title='实时交易分析',
                xaxis_title='时间',
                yaxis_title='价格',
                height=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"创建实时图表失败: {e}")
            return None
            
    def create_pattern_comparison(self,
                                patterns: List[Dict],
                                save_name: Optional[str] = None) -> str:
        """
        创建模式对比图
        
        Args:
            patterns: 模式列表
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制模式形状对比
            for i, pattern in enumerate(patterns[:4]):
                row = i // 2
                col = i % 2
                
                prices = pattern['prices']
                times = range(len(prices))
                
                axes[row, col].plot(times, prices, 'b-', label='价格')
                axes[row, col].scatter(pattern['key_points'], 
                                     [prices[i] for i in pattern['key_points']],
                                     c='r', label='关键点')
                                     
                axes[row, col].set_title(f'模式 {i+1} (置信度: {pattern["confidence"]:.2f})')
                axes[row, col].legend()
            
            # 保存图表
            if save_name is None:
                save_name = f'pattern_comparison_{datetime.now():%Y%m%d_%H%M%S}.png'
                
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"创建模式对比图失败: {e}")
            return None
            
    def create_performance_analysis(self,
                                  performance_data: Dict,
                                  save_name: Optional[str] = None) -> str:
        """
        创建性能分析图
        
        Args:
            performance_data: 性能数据
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制准确率趋势
            accuracies = performance_data['accuracy_history']
            axes[0, 0].plot(accuracies, 'b-')
            axes[0, 0].set_title('准确率趋势')
            axes[0, 0].set_ylabel('准确率')
            
            # 绘制收益分布
            returns = performance_data['returns']
            axes[0, 1].hist(returns, bins=30, color='g', alpha=0.6)
            axes[0, 1].set_title('收益分布')
            axes[0, 1].set_xlabel('收益率')
            
            # 绘制错误类型分布
            error_types = performance_data['error_types']
            error_counts = performance_data['error_counts']
            axes[1, 0].bar(error_types, error_counts, color='r', alpha=0.6)
            axes[1, 0].set_title('错误类型分布')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 绘制性能热图
            performance_matrix = performance_data['performance_matrix']
            sns.heatmap(performance_matrix, 
                       annot=True, 
                       cmap='YlOrRd',
                       ax=axes[1, 1])
            axes[1, 1].set_title('性能热图')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                save_name = f'performance_analysis_{datetime.now():%Y%m%d_%H%M%S}.png'
                
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"创建性能分析图失败: {e}")
            return None
            
    def create_error_analysis(self,
                            error_data: Dict,
                            save_name: Optional[str] = None) -> str:
        """
        创建错误分析图
        
        Args:
            error_data: 错误数据
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制错误率趋势
            error_rates = error_data['error_rates']
            axes[0, 0].plot(error_rates, 'r-')
            axes[0, 0].set_title('错误率趋势')
            axes[0, 0].set_ylabel('错误率')
            
            # 绘制错误分布
            error_values = error_data['error_values']
            axes[0, 1].hist(error_values, bins=30, color='b', alpha=0.6)
            axes[0, 1].set_title('错误分布')
            axes[0, 1].set_xlabel('错误大小')
            
            # 绘制错误相关性
            correlation_matrix = error_data['correlation_matrix']
            sns.heatmap(correlation_matrix,
                       annot=True,
                       cmap='coolwarm',
                       ax=axes[1, 0])
            axes[1, 0].set_title('错误相关性')
            
            # 绘制错误来源分析
            sources = error_data['error_sources']
            counts = error_data['source_counts']
            axes[1, 1].pie(counts,
                          labels=sources,
                          autopct='%1.1f%%',
                          colors=plt.cm.Pastel1.colors)
            axes[1, 1].set_title('错误来源分析')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                save_name = f'error_analysis_{datetime.now():%Y%m%d_%H%M%S}.png'
                
            save_path = self.save_dir / save_name
            plt.savefig(save_path)
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"创建错误分析图失败: {e}")
            return None
            
    def create_dashboard(self,
                        real_time_data: Dict,
                        analysis_data: Dict,
                        save_name: Optional[str] = None) -> str:
        """
        创建综合仪表板
        
        Args:
            real_time_data: 实时数据
            analysis_data: 分析数据
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建仪表板
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '实时K线图', 'SBS模式识别',
                    '性能指标', '错误分析',
                    '模式统计', '交易信号'
                ),
                specs=[
                    [{"type": "candlestick"}, {"type": "scatter"}],
                    [{"type": "indicator"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # 添加实时K线图
            fig.add_trace(
                go.Candlestick(
                    x=real_time_data['time'],
                    open=real_time_data['open'],
                    high=real_time_data['high'],
                    low=real_time_data['low'],
                    close=real_time_data['close']
                ),
                row=1, col=1
            )
            
            # 添加SBS模式识别结果
            for pattern in real_time_data['patterns']:
                fig.add_trace(
                    go.Scatter(
                        x=pattern['time'],
                        y=pattern['price'],
                        mode='lines+markers',
                        name=f'SBS {pattern["type"]}'
                    ),
                    row=1, col=2
                )
            
            # 添加性能指标
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=analysis_data['current_performance'],
                    delta={'reference': analysis_data['previous_performance']}
                ),
                row=2, col=1
            )
            
            # 添加错误分析
            fig.add_trace(
                go.Scatter(
                    x=analysis_data['error_time'],
                    y=analysis_data['error_values'],
                    mode='lines',
                    name='错误率'
                ),
                row=2, col=2
            )
            
            # 添加模式统计
            fig.add_trace(
                go.Bar(
                    x=analysis_data['pattern_types'],
                    y=analysis_data['pattern_counts'],
                    name='模式分布'
                ),
                row=3, col=1
            )
            
            # 添加交易信号
            fig.add_trace(
                go.Scatter(
                    x=real_time_data['signal_time'],
                    y=real_time_data['signal_price'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=real_time_data['signal_colors'],
                        symbol=real_time_data['signal_symbols']
                    ),
                    name='交易信号'
                ),
                row=3, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="交易分析仪表板"
            )
            
            # 保存仪表板
            if save_name is None:
                save_name = f'dashboard_{datetime.now():%Y%m%d_%H%M%S}.html'
                
            save_path = self.save_dir / save_name
            fig.write_html(str(save_path))
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"创建仪表板失败: {e}")
            return None 
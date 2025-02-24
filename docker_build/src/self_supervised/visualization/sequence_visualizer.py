import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.config import EVALUATION_SYSTEM

logger = logging.getLogger(__name__)

class SequenceVisualizer:
    """序列可视化器"""
    
    def __init__(self, save_dir: str = 'visualization'):
        """
        初始化可视化器
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 设置Plotly主题
        self.plotly_template = 'plotly_dark'
        
        # 设置颜色方案
        self.color_scheme = {
            'up': '#26a69a',
            'down': '#ef5350',
            'ma20': '#42a5f5',
            'ma50': '#7e57c2',
            'pivot_high': '#ffd54f',
            'pivot_low': '#aed581'
        }
        
    def plot_sequence_validation(self, sequence_data: pd.DataFrame, 
                               validation_result: Dict,
                               save_name: Optional[str] = None) -> str:
        """
        绘制序列验证结果
        
        Args:
            sequence_data: 序列数据
            validation_result: 验证结果
            save_name: 保存文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
            
            # 绘制K线图
            self._plot_candlesticks(ax1, sequence_data)
            
            # 标注关键点
            if 'points' in validation_result:
                self._plot_key_points(ax1, validation_result['points'])
            
            # 绘制成交量
            self._plot_volume(ax2, sequence_data)
            
            # 添加验证分数
            if 'score' in validation_result:
                self._add_validation_score(ax1, validation_result['score'])
            
            # 保存图表
            if save_name is None:
                save_name = f"sequence_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"序列验证图表已保存: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"绘制序列验证图表失败: {e}")
            return ""
            
    def plot_training_progress(self, history: Dict[str, List[float]],
                             save_name: Optional[str] = None) -> str:
        """
        绘制训练进度
        
        Args:
            history: 训练历史数据
            save_name: 保存文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制损失曲线
            if 'loss' in history and 'val_loss' in history:
                self._plot_loss_curves(axes[0, 0], history)
            
            # 绘制准确率曲线
            if 'accuracy' in history and 'val_accuracy' in history:
                self._plot_accuracy_curves(axes[0, 1], history)
            
            # 绘制评估指标
            if 'metrics' in history:
                self._plot_evaluation_metrics(axes[1, 0], history['metrics'])
            
            # 绘制学习率变化
            if 'learning_rate' in history:
                self._plot_learning_rate(axes[1, 1], history['learning_rate'])
            
            # 保存图表
            if save_name is None:
                save_name = f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练进度图表已保存: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"绘制训练进度图表失败: {e}")
            return ""
            
    def plot_evaluation_results(self, results: Dict,
                              save_name: Optional[str] = None) -> str:
        """
        绘制评估结果
        
        Args:
            results: 评估结果
            save_name: 保存文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制混淆矩阵
            if 'confusion_matrix' in results:
                self._plot_confusion_matrix(axes[0, 0], results['confusion_matrix'])
            
            # 绘制ROC曲线
            if 'roc_curve' in results:
                self._plot_roc_curve(axes[0, 1], results['roc_curve'])
            
            # 绘制精确率-召回率曲线
            if 'pr_curve' in results:
                self._plot_pr_curve(axes[1, 0], results['pr_curve'])
            
            # 绘制各类别F1分数
            if 'f1_scores' in results:
                self._plot_f1_scores(axes[1, 1], results['f1_scores'])
            
            # 保存图表
            if save_name is None:
                save_name = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"评估结果图表已保存: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"绘制评估结果图表失败: {e}")
            return ""
            
    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """绘制K线图"""
        # 计算上涨和下跌
        up = data[data['close'] >= data['open']]
        down = data[data['close'] < data['open']]
        
        # 绘制上涨K线
        ax.bar(up.index, up['high'] - up['low'], 
               bottom=up['low'], width=0.8, color='red', alpha=0.7)
        ax.bar(up.index, up['close'] - up['open'], 
               bottom=up['open'], width=0.8, color='red')
        
        # 绘制下跌K线
        ax.bar(down.index, down['high'] - down['low'],
               bottom=down['low'], width=0.8, color='green', alpha=0.7)
        ax.bar(down.index, down['close'] - down['open'],
               bottom=down['open'], width=0.8, color='green')
        
        ax.set_title('价格走势', fontsize=12)
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('价格', fontsize=10)
        
    def _plot_volume(self, ax, data: pd.DataFrame):
        """绘制成交量"""
        colors = ['red' if close >= open_ else 'green' 
                 for close, open_ in zip(data['close'], data['open'])]
        ax.bar(data.index, data['volume'], color=colors, alpha=0.7)
        ax.set_title('成交量', fontsize=12)
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('成交量', fontsize=10)
        
    def _plot_key_points(self, ax, points: Dict):
        """标注关键点"""
        colors = {
            'breakout': 'red',
            'point1': 'blue',
            'point2': 'green',
            'point3': 'purple',
            'point4': 'orange',
            'point5': 'brown'
        }
        
        for point_type, point_data in points.items():
            if point_data and 'index' in point_data and 'price' in point_data:
                ax.scatter(point_data['index'], point_data['price'],
                         c=colors.get(point_type, 'black'),
                         marker='*', s=200, label=point_type)
                
        ax.legend(loc='upper right')
        
    def _add_validation_score(self, ax, score: float):
        """添加验证分数"""
        threshold = EVALUATION_SYSTEM['thresholds']['min_score']
        color = 'green' if score >= threshold else 'red'
        ax.text(0.02, 0.98, f'验证分数: {score:.2f}',
                transform=ax.transAxes, color=color,
                fontsize=12, verticalalignment='top')
        
    def _plot_loss_curves(self, ax, history: Dict):
        """绘制损失曲线"""
        ax.plot(history['loss'], label='训练损失')
        ax.plot(history['val_loss'], label='验证损失')
        ax.set_title('损失曲线')
        ax.set_xlabel('轮次')
        ax.set_ylabel('损失')
        ax.legend()
        
    def _plot_accuracy_curves(self, ax, history: Dict):
        """绘制准确率曲线"""
        ax.plot(history['accuracy'], label='训练准确率')
        ax.plot(history['val_accuracy'], label='验证准确率')
        ax.set_title('准确率曲线')
        ax.set_xlabel('轮次')
        ax.set_ylabel('准确率')
        ax.legend()
        
    def _plot_evaluation_metrics(self, ax, metrics: Dict):
        """绘制评估指标"""
        x = np.arange(len(metrics))
        ax.bar(x, list(metrics.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45)
        ax.set_title('评估指标')
        ax.set_ylabel('分数')
        
    def _plot_learning_rate(self, ax, lr_history: List[float]):
        """绘制学习率变化"""
        ax.plot(lr_history)
        ax.set_title('学习率变化')
        ax.set_xlabel('步数')
        ax.set_ylabel('学习率')
        
    def _plot_confusion_matrix(self, ax, cm: np.ndarray):
        """绘制混淆矩阵"""
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title('混淆矩阵')
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        
    def _plot_roc_curve(self, ax, roc_data: Dict):
        """绘制ROC曲线"""
        ax.plot(roc_data['fpr'], roc_data['tpr'],
                label=f'ROC (AUC = {roc_data["auc"]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title('ROC曲线')
        ax.set_xlabel('假阳性率')
        ax.set_ylabel('真阳性率')
        ax.legend()
        
    def _plot_pr_curve(self, ax, pr_data: Dict):
        """绘制精确率-召回率曲线"""
        ax.plot(pr_data['recall'], pr_data['precision'],
                label=f'PR (AP = {pr_data["ap"]:.2f})')
        ax.set_title('精确率-召回率曲线')
        ax.set_xlabel('召回率')
        ax.set_ylabel('精确率')
        ax.legend()
        
    def _plot_f1_scores(self, ax, f1_scores: Dict):
        """绘制F1分数"""
        x = np.arange(len(f1_scores))
        ax.bar(x, list(f1_scores.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(list(f1_scores.keys()), rotation=45)
        ax.set_title('各类别F1分数')
        ax.set_ylabel('F1分数')
        
    def plot_interactive_chart(self, 
                             data: pd.DataFrame,
                             validation_result: Dict,
                             save_name: Optional[str] = None) -> str:
        """
        生成交互式图表
        
        Args:
            data: K线数据
            validation_result: 验证结果
            save_name: 保存文件名
            
        Returns:
            保存的HTML文件路径
        """
        try:
            # 创建图表
            fig = go.Figure()
            
            # 添加K线图
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='K线'
                )
            )
            
            # 添加移动平均线
            ma20 = data['close'].rolling(window=20).mean()
            ma50 = data['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma20,
                    name='MA20',
                    line=dict(color=self.color_scheme['ma20'])
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma50,
                    name='MA50',
                    line=dict(color=self.color_scheme['ma50'])
                )
            )
            
            # 添加支撑阻力位
            if 'market_structure' in validation_result:
                pivots = validation_result['market_structure']['pivots']
                for high in pivots['highs']:
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[high[0]]],
                            y=[high[1]],
                            mode='markers',
                            name='阻力位',
                            marker=dict(
                                color=self.color_scheme['pivot_high'],
                                size=10,
                                symbol='triangle-up'
                            )
                        )
                    )
                
                for low in pivots['lows']:
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[low[0]]],
                            y=[low[1]],
                            mode='markers',
                            name='支撑位',
                            marker=dict(
                                color=self.color_scheme['pivot_low'],
                                size=10,
                                symbol='triangle-down'
                            )
                        )
                    )
            
            # 更新布局
            fig.update_layout(
                title='序列分析结果',
                yaxis_title='价格',
                template=self.plotly_template,
                hovermode='x unified'
            )
            
            # 保存图表
            if save_name is None:
                save_name = f"sequence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
            save_path = os.path.join(self.save_dir, save_name)
            fig.write_html(save_path)
            
            return save_path
            
        except Exception as e:
            logger.error(f"生成交互式图表失败: {e}")
            return ""
            
    def plot_3d_volume_profile(self,
                              data: pd.DataFrame,
                              validation_result: Dict,
                              save_name: Optional[str] = None) -> str:
        """
        生成3D成交量分布图
        
        Args:
            data: K线数据
            validation_result: 验证结果
            save_name: 保存文件名
            
        Returns:
            保存的HTML文件路径
        """
        try:
            # 计算价格区间
            price_range = np.linspace(
                data['low'].min(),
                data['high'].max(),
                50
            )
            
            # 计算时间区间
            time_range = np.arange(len(data))
            
            # 创建网格
            price_grid, time_grid = np.meshgrid(price_range, time_range)
            volume_grid = np.zeros_like(price_grid)
            
            # 计算成交量分布
            for t in time_range:
                price_idx = np.where(
                    (price_range >= data['low'].iloc[t]) &
                    (price_range <= data['high'].iloc[t])
                )[0]
                volume_grid[t, price_idx] = data['volume'].iloc[t] / len(price_idx)
            
            # 创建3D图表
            fig = go.Figure(data=[
                go.Surface(
                    x=time_grid,
                    y=price_grid,
                    z=volume_grid,
                    colorscale='Viridis'
                )
            ])
            
            # 更新布局
            fig.update_layout(
                title='3D成交量分布',
                scene=dict(
                    xaxis_title='时间',
                    yaxis_title='价格',
                    zaxis_title='成交量'
                ),
                template=self.plotly_template
            )
            
            # 保存图表
            if save_name is None:
                save_name = f"volume_profile_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
            save_path = os.path.join(self.save_dir, save_name)
            fig.write_html(save_path)
            
            return save_path
            
        except Exception as e:
            logger.error(f"生成3D成交量分布图失败: {e}")
            return ""
            
    def create_dashboard(self,
                        data: pd.DataFrame,
                        validation_result: Dict,
                        save_name: Optional[str] = None) -> str:
        """
        创建监控面板
        
        Args:
            data: K线数据
            validation_result: 验证结果
            save_name: 保存文件名
            
        Returns:
            保存的HTML文件路径
        """
        try:
            # 创建仪表板布局
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'K线图',
                    '趋势分析',
                    '市场结构',
                    '验证得分'
                )
            )
            
            # 1. K线图
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
            
            # 2. 趋势分析
            if 'trend_strength' in validation_result:
                trend = validation_result['trend_strength']
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=trend['score'] * 100,
                        title={'text': "趋势强度"},
                        gauge={'axis': {'range': [0, 100]}}
                    ),
                    row=1, col=2
                )
            
            # 3. 市场结构
            if 'market_structure' in validation_result:
                structure = validation_result['market_structure']
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=structure['score'] * 100,
                        title={'text': "市场结构"},
                        gauge={'axis': {'range': [0, 100]}}
                    ),
                    row=2, col=1
                )
            
            # 4. 验证得分
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=validation_result['score'] * 100,
                    title={'text': "综合得分"},
                    gauge={'axis': {'range': [0, 100]}}
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                showlegend=False,
                template=self.plotly_template
            )
            
            # 保存仪表板
            if save_name is None:
                save_name = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
            save_path = os.path.join(self.save_dir, save_name)
            fig.write_html(save_path)
            
            return save_path
            
        except Exception as e:
            logger.error(f"创建监控面板失败: {e}")
            return "" 
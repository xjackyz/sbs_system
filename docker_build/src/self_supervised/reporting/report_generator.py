import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from jinja2 import Template

from src.self_supervised.visualization.sequence_visualizer import SequenceVisualizer
from config.config import EVALUATION_SYSTEM, TRAINING_WORKFLOW

logger = logging.getLogger(__name__)

class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, save_dir: str = 'reports'):
        """
        初始化报告生成器
        
        Args:
            save_dir: 报告保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.visualizer = SequenceVisualizer(os.path.join(save_dir, 'figures'))
        
    def generate_training_report(self, 
                               training_history: Dict,
                               evaluation_results: Dict,
                               model_config: Dict,
                               save_name: Optional[str] = None) -> str:
        """
        生成训练报告
        
        Args:
            training_history: 训练历史数据
            evaluation_results: 评估结果
            model_config: 模型配置
            save_name: 保存文件名
            
        Returns:
            str: 报告文件路径
        """
        try:
            # 生成可视化图表
            training_plot = self.visualizer.plot_training_progress(
                training_history,
                'training_progress.png'
            )
            evaluation_plot = self.visualizer.plot_evaluation_results(
                evaluation_results,
                'evaluation_results.png'
            )
            
            # 计算关键指标
            metrics = self._calculate_metrics(training_history, evaluation_results)
            
            # 生成报告内容
            report_content = self._generate_training_report_content(
                metrics=metrics,
                training_plot=training_plot,
                evaluation_plot=evaluation_plot,
                model_config=model_config,
                training_history=training_history
            )
            
            # 保存报告
            if save_name is None:
                save_name = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_path = os.path.join(self.save_dir, save_name)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"训练报告已生成: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"生成训练报告失败: {e}")
            return ""
            
    def generate_evaluation_report(self,
                                 validation_results: List[Dict],
                                 sequence_data: List[pd.DataFrame],
                                 save_name: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            validation_results: 验证结果列表
            sequence_data: 序列数据列表
            save_name: 保存文件名
            
        Returns:
            str: 报告文件路径
        """
        try:
            # 生成序列验证图表
            validation_plots = []
            for i, (result, data) in enumerate(zip(validation_results, sequence_data)):
                plot_path = self.visualizer.plot_sequence_validation(
                    data,
                    result,
                    f'sequence_validation_{i}.png'
                )
                validation_plots.append(plot_path)
            
            # 计算统计指标
            stats = self._calculate_validation_stats(validation_results)
            
            # 生成报告内容
            report_content = self._generate_evaluation_report_content(
                stats=stats,
                validation_plots=validation_plots,
                validation_results=validation_results
            )
            
            # 保存报告
            if save_name is None:
                save_name = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_path = os.path.join(self.save_dir, save_name)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"评估报告已生成: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")
            return ""
            
    def _calculate_metrics(self, 
                         training_history: Dict,
                         evaluation_results: Dict) -> Dict:
        """计算关键指标"""
        metrics = {
            'final_loss': training_history['loss'][-1],
            'final_val_loss': training_history['val_loss'][-1],
            'best_val_loss': min(training_history['val_loss']),
            'convergence_epoch': training_history['val_loss'].index(
                min(training_history['val_loss'])
            ),
            'total_epochs': len(training_history['loss'])
        }
        
        if 'accuracy' in training_history:
            metrics.update({
                'final_accuracy': training_history['accuracy'][-1],
                'final_val_accuracy': training_history['val_accuracy'][-1],
                'best_val_accuracy': max(training_history['val_accuracy'])
            })
            
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            metrics['overall_accuracy'] = np.trace(cm) / np.sum(cm)
            
        return metrics
        
    def _calculate_validation_stats(self, validation_results: List[Dict]) -> Dict:
        """计算验证统计信息"""
        scores = [result['score'] for result in validation_results]
        threshold = EVALUATION_SYSTEM['thresholds']['min_score']
        
        stats = {
            'total_sequences': len(validation_results),
            'valid_sequences': sum(1 for s in scores if s >= threshold),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores)
        }
        
        # 按阶段统计
        stage_stats = {}
        for result in validation_results:
            stage = result.get('stage', 'unknown')
            if stage not in stage_stats:
                stage_stats[stage] = {'count': 0, 'valid': 0}
            stage_stats[stage]['count'] += 1
            if result['score'] >= threshold:
                stage_stats[stage]['valid'] += 1
                
        stats['stage_stats'] = stage_stats
        
        return stats
        
    def _generate_training_report_content(self,
                                        metrics: Dict,
                                        training_plot: str,
                                        evaluation_plot: str,
                                        model_config: Dict,
                                        training_history: Dict) -> str:
        """生成训练报告内容"""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>训练报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .metric { margin: 10px 0; }
                img { max-width: 100%; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>训练报告</h1>
            <div class="section">
                <h2>训练概况</h2>
                <div class="metric">总轮次: {{ metrics.total_epochs }}</div>
                <div class="metric">最佳验证损失: {{ "%.4f"|format(metrics.best_val_loss) }}</div>
                <div class="metric">收敛轮次: {{ metrics.convergence_epoch }}</div>
                {% if metrics.best_val_accuracy %}
                <div class="metric">最佳验证准确率: {{ "%.2f%%"|format(metrics.best_val_accuracy * 100) }}</div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>训练配置</h2>
                <table>
                    <tr><th>参数</th><th>值</th></tr>
                    {% for key, value in model_config.items() %}
                    <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>训练过程</h2>
                <img src="{{ training_plot }}" alt="训练进度">
            </div>
            
            <div class="section">
                <h2>评估结果</h2>
                <img src="{{ evaluation_plot }}" alt="评估结果">
            </div>
            
            <div class="section">
                <h2>训练阶段</h2>
                {% for stage, info in stages.items() %}
                <h3>{{ info.name }}</h3>
                <div class="metric">目标: {{ info.objective }}</div>
                <div class="metric">持续时间: {{ info.duration }}</div>
                {% endfor %}
            </div>
        </body>
        </html>
        ''')
        
        return template.render(
            metrics=metrics,
            training_plot=os.path.relpath(training_plot, self.save_dir),
            evaluation_plot=os.path.relpath(evaluation_plot, self.save_dir),
            model_config=model_config,
            stages=TRAINING_WORKFLOW['training_stages']
        )
        
    def _generate_evaluation_report_content(self,
                                          stats: Dict,
                                          validation_plots: List[str],
                                          validation_results: List[Dict]) -> str:
        """生成评估报告内容"""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .stat { margin: 10px 0; }
                .sequence { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
                img { max-width: 100%; }
                .valid { color: green; }
                .invalid { color: red; }
            </style>
        </head>
        <body>
            <h1>评估报告</h1>
            
            <div class="section">
                <h2>总体统计</h2>
                <div class="stat">总序列数: {{ stats.total_sequences }}</div>
                <div class="stat">有效序列数: {{ stats.valid_sequences }}</div>
                <div class="stat">平均分数: {{ "%.2f"|format(stats.average_score) }}</div>
                <div class="stat">分数标准差: {{ "%.2f"|format(stats.score_std) }}</div>
                <div class="stat">最低分数: {{ "%.2f"|format(stats.min_score) }}</div>
                <div class="stat">最高分数: {{ "%.2f"|format(stats.max_score) }}</div>
            </div>
            
            <div class="section">
                <h2>阶段统计</h2>
                {% for stage, stage_stat in stats.stage_stats.items() %}
                <h3>{{ stage }}</h3>
                <div class="stat">序列数: {{ stage_stat.count }}</div>
                <div class="stat">有效序列数: {{ stage_stat.valid }}</div>
                <div class="stat">有效率: {{ "%.2f%%"|format(stage_stat.valid / stage_stat.count * 100) }}</div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>序列验证结果</h2>
                {% for plot, result in zip(validation_plots, validation_results) %}
                <div class="sequence">
                    <h3 class="{{ 'valid' if result.score >= threshold else 'invalid' }}">
                        序列 #{{ loop.index }} (得分: {{ "%.2f"|format(result.score) }})
                    </h3>
                    <div>阶段: {{ result.stage }}</div>
                    <img src="{{ plot }}" alt="序列验证图表">
                    <div>详细信息:</div>
                    <pre>{{ result.details | tojson(indent=2) }}</pre>
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        ''')
        
        return template.render(
            stats=stats,
            validation_plots=[os.path.relpath(p, self.save_dir) for p in validation_plots],
            validation_results=validation_results,
            threshold=EVALUATION_SYSTEM['thresholds']['min_score'],
            zip=zip
        ) 
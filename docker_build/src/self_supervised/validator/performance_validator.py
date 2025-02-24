import os
import json
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.utils.logger import setup_logger

logger = setup_logger('performance_validator')

class PerformanceValidator:
    """自监督学习性能验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.tasks_dir = 'training_data/tasks/'
        self.results_dir = 'validation_results/'
        
        # 确保目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 性能指标定义
        self.metrics = {
            'continuity': {
                'point_accuracy': '点位预测准确率',
                'time_error': '时间误差',
                'price_error': '价格误差'
            },
            'completeness': {
                'classification_accuracy': '分类准确率',
                'precision': '精确率',
                'recall': '召回率'
            },
            'consistency': {
                'pattern_similarity': '模式相似度',
                'time_alignment': '时间对齐度',
                'structure_consistency': '结构一致性'
            }
        }
        
        # 设置matplotlib样式
        plt.style.use('default')
    
    def validate_performance(self, predictions: List[Dict]) -> Dict:
        """验证模型性能"""
        logger.info("开始验证模型性能")
        
        try:
            # 按任务类型分组验证
            results = {}
            
            # 验证连续性预测任务
            continuity_preds = [p for p in predictions if p['task_type'] == 'continuity']
            if continuity_preds:
                results['continuity'] = self._validate_continuity(continuity_preds)
            
            # 验证完整性判断任务
            completeness_preds = [p for p in predictions if p['task_type'] == 'completeness']
            if completeness_preds:
                results['completeness'] = self._validate_completeness(completeness_preds)
            
            # 验证一致性验证任务
            consistency_preds = [p for p in predictions if p['task_type'] == 'consistency']
            if consistency_preds:
                results['consistency'] = self._validate_consistency(consistency_preds)
            
            # 生成验证报告
            self._generate_validation_report(results)
            
            # 可视化结果
            self._visualize_results(results)
            
            logger.info("性能验证完成")
            return results
            
        except Exception as e:
            logger.error(f"性能验证出错: {e}")
            return {}
    
    def _validate_continuity(self, predictions: List[Dict]) -> Dict:
        """验证连续性预测性能"""
        try:
            results = {
                'point_accuracy': 0.0,
                'time_errors': [],
                'price_errors': [],
                'by_point_type': {}
            }
            
            for pred in predictions:
                # 获取真实值和预测值
                true_point = pred['true_value']
                pred_point = pred['prediction']
                
                # 计算点位准确率
                if self._is_point_correct(true_point, pred_point):
                    results['point_accuracy'] += 1
                
                # 计算时间误差
                time_error = self._calculate_time_error(true_point, pred_point)
                results['time_errors'].append(time_error)
                
                # 计算价格误差
                price_error = self._calculate_price_error(true_point, pred_point)
                results['price_errors'].append(price_error)
                
                # 按点位类型统计
                point_type = true_point['type']
                if point_type not in results['by_point_type']:
                    results['by_point_type'][point_type] = {
                        'count': 0,
                        'correct': 0
                    }
                results['by_point_type'][point_type]['count'] += 1
                if self._is_point_correct(true_point, pred_point):
                    results['by_point_type'][point_type]['correct'] += 1
            
            # 计算最终指标
            total_predictions = len(predictions)
            if total_predictions > 0:
                results['point_accuracy'] /= total_predictions
                results['mean_time_error'] = np.mean(results['time_errors'])
                results['mean_price_error'] = np.mean(results['price_errors'])
                
                # 计算每种点位的准确率
                for point_type in results['by_point_type']:
                    stats = results['by_point_type'][point_type]
                    stats['accuracy'] = stats['correct'] / stats['count']
            
            return results
            
        except Exception as e:
            logger.error(f"验证连续性预测性能出错: {e}")
            return {}
    
    def _validate_completeness(self, predictions: List[Dict]) -> Dict:
        """验证完整性判断性能"""
        try:
            results = {
                'classification_metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                },
                'confusion_matrix': None,
                'by_sequence_type': {}
            }
            
            # 收集真实值和预测值
            y_true = []
            y_pred = []
            
            for pred in predictions:
                y_true.append(pred['true_value'])
                y_pred.append(pred['prediction'])
                
                # 按序列类型统计
                seq_type = pred['sequence_type']
                if seq_type not in results['by_sequence_type']:
                    results['by_sequence_type'][seq_type] = {
                        'total': 0,
                        'correct': 0
                    }
                results['by_sequence_type'][seq_type]['total'] += 1
                if pred['true_value'] == pred['prediction']:
                    results['by_sequence_type'][seq_type]['correct'] += 1
            
            # 计算混淆矩阵
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # 计算分类指标
            results['classification_metrics'] = self._calculate_classification_metrics(y_true, y_pred)
            
            # 计算每种序列类型的准确率
            for seq_type in results['by_sequence_type']:
                stats = results['by_sequence_type'][seq_type]
                stats['accuracy'] = stats['correct'] / stats['total']
            
            return results
            
        except Exception as e:
            logger.error(f"验证完整性判断性能出错: {e}")
            return {}
    
    def _validate_consistency(self, predictions: List[Dict]) -> Dict:
        """验证一致性验证性能"""
        try:
            results = {
                'pattern_similarity': [],
                'time_alignment': [],
                'structure_consistency': [],
                'by_sequence_type': {}
            }
            
            for pred in predictions:
                # 计算模式相似度
                similarity = self._calculate_pattern_similarity(
                    pred['sequence1'],
                    pred['sequence2']
                )
                results['pattern_similarity'].append(similarity)
                
                # 计算时间对齐度
                alignment = self._calculate_time_alignment(
                    pred['sequence1'],
                    pred['sequence2']
                )
                results['time_alignment'].append(alignment)
                
                # 计算结构一致性
                consistency = self._calculate_structure_consistency(
                    pred['sequence1'],
                    pred['sequence2']
                )
                results['structure_consistency'].append(consistency)
                
                # 按序列类型统计
                seq_type = pred['sequence_type']
                if seq_type not in results['by_sequence_type']:
                    results['by_sequence_type'][seq_type] = {
                        'pattern_similarity': [],
                        'time_alignment': [],
                        'structure_consistency': []
                    }
                stats = results['by_sequence_type'][seq_type]
                stats['pattern_similarity'].append(similarity)
                stats['time_alignment'].append(alignment)
                stats['structure_consistency'].append(consistency)
            
            # 计算平均指标
            results['mean_pattern_similarity'] = np.mean(results['pattern_similarity'])
            results['mean_time_alignment'] = np.mean(results['time_alignment'])
            results['mean_structure_consistency'] = np.mean(results['structure_consistency'])
            
            # 计算每种序列类型的平均指标
            for seq_type in results['by_sequence_type']:
                stats = results['by_sequence_type'][seq_type]
                stats['mean_pattern_similarity'] = np.mean(stats['pattern_similarity'])
                stats['mean_time_alignment'] = np.mean(stats['time_alignment'])
                stats['mean_structure_consistency'] = np.mean(stats['structure_consistency'])
            
            return results
            
        except Exception as e:
            logger.error(f"验证一致性验证性能出错: {e}")
            return {}
    
    def _is_point_correct(self, true_point: Dict, pred_point: Dict) -> bool:
        """判断点位预测是否正确"""
        # 可以根据具体需求调整判断标准
        time_error = self._calculate_time_error(true_point, pred_point)
        price_error = self._calculate_price_error(true_point, pred_point)
        
        return time_error <= 5 and price_error <= 0.001  # 示例阈值
    
    def _calculate_time_error(self, true_point: Dict, pred_point: Dict) -> float:
        """计算时间误差（单位：分钟）"""
        true_time = datetime.strptime(true_point['time'], "%Y-%m-%d %H:%M:%S")
        pred_time = datetime.strptime(pred_point['time'], "%Y-%m-%d %H:%M:%S")
        
        return abs((true_time - pred_time).total_seconds() / 60)
    
    def _calculate_price_error(self, true_point: Dict, pred_point: Dict) -> float:
        """计算价格误差（相对误差）"""
        true_price = float(true_point['price'])
        pred_price = float(pred_point['price'])
        
        return abs(true_price - pred_price) / true_price
    
    def _calculate_classification_metrics(self, y_true: List, y_pred: List) -> Dict:
        """计算分类指标"""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # 计算准确率
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        metrics['accuracy'] = correct / len(y_true)
        
        # 计算精确率和召回率
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        metrics['precision'] = np.mean(precision)
        metrics['recall'] = np.mean(recall)
        
        # 计算F1分数
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                (metrics['precision'] + metrics['recall'])
        
        return metrics
    
    def _calculate_pattern_similarity(self, seq1: Dict, seq2: Dict) -> float:
        """计算模式相似度"""
        # 可以根据具体需求实现相似度计算方法
        # 这里使用一个简单的示例实现
        points1 = set(seq1['points'].keys())
        points2 = set(seq2['points'].keys())
        
        return len(points1.intersection(points2)) / len(points1.union(points2))
    
    def _calculate_time_alignment(self, seq1: Dict, seq2: Dict) -> float:
        """计算时间对齐度"""
        # 可以根据具体需求实现时间对齐度计算方法
        # 这里使用一个简单的示例实现
        time_diffs = []
        for point in seq1['points'].keys():
            if point in seq2['points']:
                time1 = datetime.strptime(seq1['points'][point]['time'], "%Y-%m-%d %H:%M:%S")
                time2 = datetime.strptime(seq2['points'][point]['time'], "%Y-%m-%d %H:%M:%S")
                time_diff = abs((time1 - time2).total_seconds() / 60)
                time_diffs.append(time_diff)
        
        return 1 / (1 + np.mean(time_diffs)) if time_diffs else 0
    
    def _calculate_structure_consistency(self, seq1: Dict, seq2: Dict) -> float:
        """计算结构一致性"""
        # 可以根据具体需求实现结构一致性计算方法
        # 这里使用一个简单的示例实现
        if not seq1['points'] or not seq2['points']:
            return 0
            
        # 比较点位之间的相对位置关系
        consistency = 0
        total_comparisons = 0
        
        points1 = list(seq1['points'].keys())
        points2 = list(seq2['points'].keys())
        
        for i in range(len(points1)-1):
            for j in range(i+1, len(points1)):
                point1_i = seq1['points'][points1[i]]
                point1_j = seq1['points'][points1[j]]
                
                if points1[i] in points2 and points1[j] in points2:
                    point2_i = seq2['points'][points1[i]]
                    point2_j = seq2['points'][points1[j]]
                    
                    # 比较相对位置关系是否一致
                    relation1 = float(point1_j['price']) - float(point1_i['price'])
                    relation2 = float(point2_j['price']) - float(point2_i['price'])
                    
                    if (relation1 * relation2) > 0:  # 相对位置关系一致
                        consistency += 1
                    total_comparisons += 1
        
        return consistency / total_comparisons if total_comparisons > 0 else 0
    
    def _generate_validation_report(self, results: Dict):
        """生成验证报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.results_dir, f"validation_report_{timestamp}.txt")
            
            with open(report_path, 'w') as f:
                f.write("=== 自监督学习性能验证报告 ===\n")
                f.write(f"生成时间: {datetime.now()}\n\n")
                
                # 连续性预测结果
                if 'continuity' in results:
                    f.write("=== 连续性预测性能 ===\n")
                    cont_results = results['continuity']
                    f.write(f"点位预测准确率: {cont_results['point_accuracy']:.2%}\n")
                    f.write(f"平均时间误差: {cont_results['mean_time_error']:.2f} 分钟\n")
                    f.write(f"平均价格误差: {cont_results['mean_price_error']:.2%}\n\n")
                    
                    f.write("按点位类型的性能:\n")
                    for point_type, stats in cont_results['by_point_type'].items():
                        f.write(f"{point_type}:\n")
                        f.write(f"  样本数: {stats['count']}\n")
                        f.write(f"  准确率: {stats['accuracy']:.2%}\n")
                    f.write("\n")
                
                # 完整性判断结果
                if 'completeness' in results:
                    f.write("=== 完整性判断性能 ===\n")
                    comp_results = results['completeness']
                    metrics = comp_results['classification_metrics']
                    f.write(f"准确率: {metrics['accuracy']:.2%}\n")
                    f.write(f"精确率: {metrics['precision']:.2%}\n")
                    f.write(f"召回率: {metrics['recall']:.2%}\n")
                    f.write(f"F1分数: {metrics['f1_score']:.2%}\n\n")
                    
                    f.write("按序列类型的性能:\n")
                    for seq_type, stats in comp_results['by_sequence_type'].items():
                        f.write(f"{seq_type}:\n")
                        f.write(f"  样本数: {stats['total']}\n")
                        f.write(f"  准确率: {stats['accuracy']:.2%}\n")
                    f.write("\n")
                
                # 一致性验证结果
                if 'consistency' in results:
                    f.write("=== 一致性验证性能 ===\n")
                    cons_results = results['consistency']
                    f.write(f"平均模式相似度: {cons_results['mean_pattern_similarity']:.2%}\n")
                    f.write(f"平均时间对齐度: {cons_results['mean_time_alignment']:.2%}\n")
                    f.write(f"平均结构一致性: {cons_results['mean_structure_consistency']:.2%}\n\n")
                    
                    f.write("按序列类型的性能:\n")
                    for seq_type, stats in cons_results['by_sequence_type'].items():
                        f.write(f"{seq_type}:\n")
                        f.write(f"  模式相似度: {stats['mean_pattern_similarity']:.2%}\n")
                        f.write(f"  时间对齐度: {stats['mean_time_alignment']:.2%}\n")
                        f.write(f"  结构一致性: {stats['mean_structure_consistency']:.2%}\n")
            
            logger.info(f"验证报告已生成: {report_path}")
            
        except Exception as e:
            logger.error(f"生成验证报告出错: {e}")
    
    def _visualize_results(self, results: Dict):
        """可视化验证结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 连续性预测结果可视化
            if 'continuity' in results:
                self._plot_continuity_results(results['continuity'], timestamp)
            
            # 完整性判断结果可视化
            if 'completeness' in results:
                self._plot_completeness_results(results['completeness'], timestamp)
            
            # 一致性验证结果可视化
            if 'consistency' in results:
                self._plot_consistency_results(results['consistency'], timestamp)
            
        except Exception as e:
            logger.error(f"可视化结果出错: {e}")
    
    def _plot_continuity_results(self, results: Dict, timestamp: str):
        """绘制连续性预测结果图表"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 点位准确率柱状图
            point_types = list(results['by_point_type'].keys())
            accuracies = [stats['accuracy'] for stats in results['by_point_type'].values()]
            
            axes[0, 0].bar(point_types, accuracies)
            axes[0, 0].set_title('点位预测准确率')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_ylabel('准确率')
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 时间误差直方图
            axes[0, 1].hist(results['time_errors'], bins=20)
            axes[0, 1].set_title('时间误差分布')
            axes[0, 1].set_xlabel('时间误差（分钟）')
            axes[0, 1].set_ylabel('频次')
            
            # 价格误差直方图
            axes[1, 0].hist(results['price_errors'], bins=20)
            axes[1, 0].set_title('价格误差分布')
            axes[1, 0].set_xlabel('相对价格误差')
            axes[1, 0].set_ylabel('频次')
            
            # 样本数量饼图
            sizes = [stats['count'] for stats in results['by_point_type'].values()]
            axes[1, 1].pie(sizes, labels=point_types, autopct='%1.1f%%')
            axes[1, 1].set_title('样本分布')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'continuity_results_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制连续性预测结果图表出错: {e}")
    
    def _plot_completeness_results(self, results: Dict, timestamp: str):
        """绘制完整性判断结果图表"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 混淆矩阵热力图
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=axes[0, 0])
            axes[0, 0].set_title('混淆矩阵')
            
            # 分类指标柱状图
            metrics = results['classification_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[0, 1].bar(metric_names, metric_values)
            axes[0, 1].set_title('分类性能指标')
            axes[0, 1].set_ylim(0, 1)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # 序列类型准确率柱状图
            seq_types = list(results['by_sequence_type'].keys())
            accuracies = [stats['accuracy'] for stats in results['by_sequence_type'].values()]
            
            axes[1, 0].bar(seq_types, accuracies)
            axes[1, 0].set_title('序列类型准确率')
            axes[1, 0].set_ylim(0, 1)
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 样本分布饼图
            sizes = [stats['total'] for stats in results['by_sequence_type'].values()]
            axes[1, 1].pie(sizes, labels=seq_types, autopct='%1.1f%%')
            axes[1, 1].set_title('样本分布')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'completeness_results_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制完整性判断结果图表出错: {e}")
    
    def _plot_consistency_results(self, results: Dict, timestamp: str):
        """绘制一致性验证结果图表"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 整体性能指标柱状图
            metric_names = ['模式相似度', '时间对齐度', '结构一致性']
            metric_values = [
                results['mean_pattern_similarity'],
                results['mean_time_alignment'],
                results['mean_structure_consistency']
            ]
            
            axes[0, 0].bar(metric_names, metric_values)
            axes[0, 0].set_title('整体性能指标')
            axes[0, 0].set_ylim(0, 1)
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 序列类型性能对比
            seq_types = list(results['by_sequence_type'].keys())
            x = np.arange(len(seq_types))
            width = 0.25
            
            pattern_sim = [stats['mean_pattern_similarity'] for stats in results['by_sequence_type'].values()]
            time_align = [stats['mean_time_alignment'] for stats in results['by_sequence_type'].values()]
            struct_cons = [stats['mean_structure_consistency'] for stats in results['by_sequence_type'].values()]
            
            axes[0, 1].bar(x - width, pattern_sim, width, label='模式相似度')
            axes[0, 1].bar(x, time_align, width, label='时间对齐度')
            axes[0, 1].bar(x + width, struct_cons, width, label='结构一致性')
            
            axes[0, 1].set_title('序列类型性能对比')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(seq_types)
            axes[0, 1].legend()
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # 模式相似度分布直方图
            axes[1, 0].hist(results['pattern_similarity'], bins=20)
            axes[1, 0].set_title('模式相似度分布')
            axes[1, 0].set_xlabel('相似度')
            axes[1, 0].set_ylabel('频次')
            
            # 结构一致性分布直方图
            axes[1, 1].hist(results['structure_consistency'], bins=20)
            axes[1, 1].set_title('结构一致性分布')
            axes[1, 1].set_xlabel('一致性')
            axes[1, 1].set_ylabel('频次')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'consistency_results_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制一致性验证结果图表出错: {e}") 
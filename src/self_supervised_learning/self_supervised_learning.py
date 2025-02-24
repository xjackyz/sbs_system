import os
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from src.model.llava_analyzer import LLaVAAnalyzer
from src.utils.logger import setup_logger
import shutil

# 设置日志记录器
logger = setup_logger('self_supervised_learning')

class SBSSequenceLearner:
    """SBS序列自监督学习器"""
    
    def __init__(self):
        """初始化学习器"""
        self.data_dir = '/home/easyai/桌面/nq/NQ_full_1min_continuous.csv'
        self.chart_dir = 'screenshots/sequence_charts/'
        self.analysis_dir = 'analysis_results/'
        
        # 确保目录存在
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # 初始化LLaVA分析器
        self.llava_analyzer = LLaVAAnalyzer()
        
        # 序列特征配置
        self.sequence_features = {
            'breakout': {
                'min_volume_ratio': 1.5,  # 突破成交量要求
                'price_range_ratio': 0.01  # 价格范围要求
            },
            'sequence_points': {
                'point1_range': (3, 10),   # 点1形成的K线范围
                'point2_range': (5, 20),   # 点2形成的K线范围
                'point3_range': (8, 25),   # 点3形成的K线范围
                'point4_range': (10, 30),  # 点4形成的K线范围
                'point5_range': (15, 40)   # 点5形成的K线范围
            },
            'validity_checks': {
                'volume_support': True,     # 成交量支持
                'structure_integrity': True, # 结构完整性
                'trend_consistency': True   # 趋势一致性
            }
        }
    
    def prepare_sequence_data(self, start_date, end_date):
        """准备序列数据"""
        logger.info(f"准备数据 从 {start_date} 到 {end_date}")
        
        # 读取数据
        data = pd.read_csv(self.data_dir)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        # 过滤日期范围
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data.loc[mask]
        
        # 计算技术指标
        data['SMA20'] = data['close'].rolling(window=20).mean()
        data['SMA200'] = data['close'].rolling(window=200).mean()
        data['volume_ma20'] = data['volume'].rolling(window=20).mean()
        
        logger.info(f"数据准备完成，共 {len(data)} 根K线")
        return data
    
    def generate_sequence_charts(self, data, window_size=100):
        """生成序列图表"""
        logger.info("开始生成序列图表")
        
        # 清空旧图表
        for file in os.listdir(self.chart_dir):
            os.remove(os.path.join(self.chart_dir, file))
        
        # 滑动窗口生成图表
        for i in range(0, len(data) - window_size, window_size // 2):  # 50%重叠
            segment = data.iloc[i:i + window_size]
            
            # 生成图表
            filename = os.path.join(self.chart_dir, f'sequence_{i}.png')
            self._plot_sequence_chart(segment, filename)
            
            # 保存序列信息
            self._save_sequence_info(segment, filename)
            
        logger.info("序列图表生成完成")
    
    def _plot_sequence_chart(self, data, filename):
        """绘制序列图表"""
        # 设置样式
        mc = mpf.make_marketcolors(up='lightgray', down='darkgray',
                                 edge='inherit',
                                 wick='inherit',
                                 volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # 添加均线
        apds = [
            mpf.make_addplot(data['SMA20'], color='lightgray'),
            mpf.make_addplot(data['SMA200'], color='darkgray')
        ]
        
        # 绘制图表
        mpf.plot(data, type='candle',
                style=s,
                addplot=apds,
                volume=True,
                savefig=filename)
    
    def _save_sequence_info(self, data, chart_file):
        """保存序列信息"""
        info_file = chart_file.replace('.png', '_info.txt')
        
        with open(info_file, 'w') as f:
            f.write("=== 序列基本信息 ===\n")
            f.write(f"时间范围: {data.index[0]} 到 {data.index[-1]}\n")
            f.write(f"K线数量: {len(data)}\n")
            f.write(f"价格范围: {data['low'].min()} - {data['high'].max()}\n")
            f.write(f"平均成交量: {data['volume'].mean():.2f}\n")
            
            # 添加技术指标信息
            f.write("\n=== 技术指标 ===\n")
            f.write(f"SMA20范围: {data['SMA20'].min():.2f} - {data['SMA20'].max():.2f}\n")
            f.write(f"SMA200范围: {data['SMA200'].min():.2f} - {data['SMA200'].max():.2f}\n")
            f.write(f"成交量MA20范围: {data['volume_ma20'].min():.2f} - {data['volume_ma20'].max():.2f}\n")
    
    def analyze_sequences(self):
        """分析序列特征"""
        logger.info("开始分析序列特征")
        
        results = []
        for filename in os.listdir(self.chart_dir):
            if not filename.endswith('.png'):
                continue
                
            image_path = os.path.join(self.chart_dir, filename)
            try:
                # 使用LLaVA分析
                analysis = self.llava_analyzer.analyze_chart(image_path)
                
                # 提取序列特征
                features = self._extract_sequence_features(analysis)
                
                # 评估序列质量
                quality_score = self._evaluate_sequence_quality(features)
                
                # 保存分析结果
                result = {
                    'filename': filename,
                    'features': features,
                    'quality_score': quality_score,
                    'analysis': analysis
                }
                results.append(result)
                
                # 保存详细分析结果
                self._save_analysis_result(result)
                
                logger.info(f"序列分析完成: {filename}, 质量分数: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"分析序列出错 {filename}: {e}")
    
    def _extract_sequence_features(self, analysis):
        """提取序列特征"""
        sequence = analysis.get('sequence', {})
        return {
            'type': sequence.get('type'),
            'points': sequence.get('points', {}),
            'validity': sequence.get('validity', {}),
            'confidence': analysis.get('trade_signal', {}).get('confidence', 0)
        }
    
    def _evaluate_sequence_quality(self, features):
        """评估序列质量"""
        score = 0.0
        
        # 检查序列类型
        if features['type'] in ['upward', 'downward']:
            score += 0.3
        
        # 检查关键点位
        points = features['points']
        if len(points) >= 4:  # 至少有4个关键点
            score += 0.3
        
        # 检查有效性
        validity = features['validity']
        if validity.get('volume_support'):
            score += 0.2
        if not validity.get('structure_break'):
            score += 0.2
            
        # 考虑置信度
        score *= features['confidence']
        
        return score
    
    def _save_analysis_result(self, result):
        """保存分析结果"""
        filename = result['filename'].replace('.png', '_analysis.txt')
        filepath = os.path.join(self.analysis_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=== 序列分析结果 ===\n")
            f.write(f"文件名: {result['filename']}\n")
            f.write(f"质量分数: {result['quality_score']:.2f}\n\n")
            
            f.write("=== 序列特征 ===\n")
            features = result['features']
            f.write(f"类型: {features['type']}\n")
            f.write(f"置信度: {features['confidence']:.2f}\n")
            
            f.write("\n关键点位:\n")
            for point, value in features['points'].items():
                f.write(f"{point}: {value}\n")
            
            f.write("\n有效性检查:\n")
            for check, value in features['validity'].items():
                f.write(f"{check}: {value}\n")
    
    def generate_learning_report(self):
        """生成学习报告"""
        logger.info("生成学习报告")
        
        report = {
            'total_sequences': 0,
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'sequence_types': {'upward': 0, 'downward': 0, 'invalid': 0},
            'average_quality': 0.0
        }
        
        # 统计分析结果
        quality_scores = []
        for filename in os.listdir(self.analysis_dir):
            if not filename.endswith('_analysis.txt'):
                continue
                
            report['total_sequences'] += 1
            filepath = os.path.join(self.analysis_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
                
                # 提取质量分数
                for line in content.split('\n'):
                    if '质量分数:' in line:
                        score = float(line.split(':')[1])
                        quality_scores.append(score)
                        
                        if score >= 0.8:
                            report['high_quality'] += 1
                        elif score >= 0.5:
                            report['medium_quality'] += 1
                        else:
                            report['low_quality'] += 1
                
                # 统计序列类型
                if 'upward' in content:
                    report['sequence_types']['upward'] += 1
                elif 'downward' in content:
                    report['sequence_types']['downward'] += 1
                else:
                    report['sequence_types']['invalid'] += 1
        
        # 计算平均质量
        if quality_scores:
            report['average_quality'] = sum(quality_scores) / len(quality_scores)
        
        # 保存报告
        report_path = os.path.join(self.analysis_dir, 'learning_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== SBS序列自监督学习报告 ===\n")
            f.write(f"生成时间: {datetime.now()}\n\n")
            
            f.write("=== 数量统计 ===\n")
            f.write(f"总序列数: {report['total_sequences']}\n")
            f.write(f"高质量序列: {report['high_quality']}\n")
            f.write(f"中等质量序列: {report['medium_quality']}\n")
            f.write(f"低质量序列: {report['low_quality']}\n\n")
            
            f.write("=== 序列类型分布 ===\n")
            for type_name, count in report['sequence_types'].items():
                f.write(f"{type_name}: {count}\n")
            
            f.write(f"\n平均质量分数: {report['average_quality']:.2f}\n")
        
        logger.info(f"学习报告已生成: {report_path}")
        return report

def main():
    """主函数"""
    try:
        # 创建学习器
        learner = SBSSequenceLearner()
        
        # 设置日期范围
        start_date = '2024-12-01'
        end_date = '2024-12-31'
        
        # 准备数据
        data = learner.prepare_sequence_data(start_date, end_date)
        
        # 生成序列图表
        learner.generate_sequence_charts(data)
        
        # 分析序列
        learner.analyze_sequences()
        
        # 生成报告
        report = learner.generate_learning_report()
        
        logger.info("自监督学习完成")
        
    except Exception as e:
        logger.error(f"自监督学习过程出错: {e}")
        return

if __name__ == "__main__":
    main() 
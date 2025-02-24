import os
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import cv2
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.utils.logger import setup_logger

logger = setup_logger('sequence_generator')

class SequenceGenerator:
    """SBS序列数据生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.data_dir = '/home/easyai/桌面/nq/NQ_full_1min_continuous.csv'
        self.output_dir = 'training_data/sequences/'
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 序列特征定义
        self.sequence_rules = {
            'upward': {
                'breakout': {
                    'volume_ratio': 1.5,     # 突破成交量比例
                    'price_change': 0.001    # 最小价格变化
                },
                'points': {
                    'point1': {'range': (3, 10), 'retrace': (0.382, 0.618)},  # 回调范围
                    'point2': {'range': (5, 20), 'extension': (1.0, 1.618)},  # 扩展目标
                    'point3': {'range': (8, 25), 'retrace': (0.5, 0.786)},    # 回调目标
                    'point4': {'range': (10, 30), 'support': 'point1'},       # 支撑位置
                    'point5': {'range': (15, 40), 'target': 'point2'}         # 目标位置
                }
            },
            'downward': {
                'breakout': {
                    'volume_ratio': 1.5,
                    'price_change': 0.001
                },
                'points': {
                    'point1': {'range': (3, 10), 'retrace': (0.382, 0.618)},
                    'point2': {'range': (5, 20), 'extension': (1.0, 1.618)},
                    'point3': {'range': (8, 25), 'retrace': (0.5, 0.786)},
                    'point4': {'range': (10, 30), 'resistance': 'point1'},
                    'point5': {'range': (15, 40), 'target': 'point2'}
                }
            }
        }
        
        # 数据增强配置
        self.augmentation_config = {
            'time_shift': (-5, 5),       # K线时移范围
            'noise_level': (0.0001, 0.001),  # 价格噪声水平
            'volume_scale': (0.8, 1.2)    # 成交量缩放范围
        }
        
        self.num_cores = mp.cpu_count()
        logger.info(f"序列生成器初始化，使用 {self.num_cores} 个CPU核心")
    
    def generate_training_data(self, start_date: str, end_date: str, window_size: int = 100):
        """并行生成训练数据"""
        try:
            # 加载数据
            data = self._load_data(start_date, end_date)
            if data is None:
                return []
                
            # 将数据分成多个块进行并行处理
            chunk_size = len(data) // self.num_cores
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            # 使用进程池并行处理
            sequences = []
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                futures = [
                    executor.submit(self._process_chunk, chunk, window_size)
                    for chunk in chunks
                ]
                for future in futures:
                    sequences.extend(future.result())
            
            logger.info(f"并行生成了 {len(sequences)} 个序列")
            return sequences
            
        except Exception as e:
            logger.error(f"生成训练数据出错: {e}")
            return []
            
    def _process_chunk(self, data: pd.DataFrame, window_size: int) -> List[Dict]:
        """处理数据块"""
        sequences = []
        try:
            # 识别序列
            identified_sequences = self._identify_sequences(data)
            
            # 生成样本
            for i, sequence in enumerate(identified_sequences):
                sample = self._generate_sequence_sample(sequence, i)
                if sample:
                    sequences.append(sample)
                    
            return sequences
            
        except Exception as e:
            logger.error(f"处理数据块出错: {e}")
            return []
    
    def _load_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载数据"""
        try:
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
            
            logger.info(f"数据加载完成，共 {len(data)} 根K线")
            return data
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None
    
    def _identify_sequences(self, data: pd.DataFrame) -> List[Dict]:
        """识别潜在的SBS序列"""
        sequences = []
        
        # 遍历数据寻找突破点
        for i in range(20, len(data)-40):  # 确保有足够的前后数据
            try:
                # 检查上升序列
                if self._check_upward_breakout(data, i):
                    seq = self._validate_sequence(data, i, 'upward')
                    if seq:
                        sequences.append(seq)
                
                # 检查下降序列
                elif self._check_downward_breakout(data, i):
                    seq = self._validate_sequence(data, i, 'downward')
                    if seq:
                        sequences.append(seq)
                        
            except Exception as e:
                logger.error(f"序列识别出错 at {data.index[i]}: {e}")
                continue
        
        return sequences
    
    def _check_upward_breakout(self, data: pd.DataFrame, index: int) -> bool:
        """检查上升突破"""
        rules = self.sequence_rules['upward']['breakout']
        
        # 检查价格突破
        price_change = (data['close'].iloc[index] - data['close'].iloc[index-1]) / data['close'].iloc[index-1]
        if price_change < rules['price_change']:
            return False
            
        # 检查成交量
        volume_ratio = data['volume'].iloc[index] / data['volume_ma20'].iloc[index]
        if volume_ratio < rules['volume_ratio']:
            return False
            
        # 检查趋势
        if data['close'].iloc[index] < data['SMA20'].iloc[index]:
            return False
            
        return True
    
    def _check_downward_breakout(self, data: pd.DataFrame, index: int) -> bool:
        """检查下降突破"""
        rules = self.sequence_rules['downward']['breakout']
        
        # 检查价格突破
        price_change = (data['close'].iloc[index-1] - data['close'].iloc[index]) / data['close'].iloc[index-1]
        if price_change < rules['price_change']:
            return False
            
        # 检查成交量
        volume_ratio = data['volume'].iloc[index] / data['volume_ma20'].iloc[index]
        if volume_ratio < rules['volume_ratio']:
            return False
            
        # 检查趋势
        if data['close'].iloc[index] > data['SMA20'].iloc[index]:
            return False
            
        return True
    
    def _validate_sequence(self, data: pd.DataFrame, breakout_index: int, seq_type: str) -> Optional[Dict]:
        """验证并提取完整序列"""
        rules = self.sequence_rules[seq_type]['points']
        sequence = {
            'type': seq_type,
            'breakout_index': breakout_index,
            'data': data,
            'points': {}
        }
        
        try:
            # 验证点1（回调点）
            point1 = self._find_point1(data, breakout_index, seq_type)
            if not point1:
                return None
            sequence['points']['point1'] = point1
            
            # 验证点2（极值点）
            point2 = self._find_point2(data, point1['index'], seq_type)
            if not point2:
                return None
            sequence['points']['point2'] = point2
            
            # 验证点3（回调点）
            point3 = self._find_point3(data, point2['index'], point1, seq_type)
            if not point3:
                return None
            sequence['points']['point3'] = point3
            
            # 验证点4（确认点）
            point4 = self._find_point4(data, point3['index'], point1, seq_type)
            if not point4:
                return None
            sequence['points']['point4'] = point4
            
            # 验证点5（目标点）
            point5 = self._find_point5(data, point4['index'], point2, seq_type)
            if point5:
                sequence['points']['point5'] = point5
            
            return sequence
            
        except Exception as e:
            logger.error(f"序列验证出错: {e}")
            return None
    
    def _find_point1(self, data: pd.DataFrame, breakout_index: int, seq_type: str) -> Optional[Dict]:
        """寻找点1"""
        rules = self.sequence_rules[seq_type]['points']['point1']
        start_idx = breakout_index + rules['range'][0]
        end_idx = min(breakout_index + rules['range'][1], len(data)-1)
        
        if seq_type == 'upward':
            # 寻找回调低点
            point1_idx = data['low'].iloc[start_idx:end_idx].astype(float).idxmin()
            point1_idx_int = data.index.get_loc(point1_idx)
            retrace = (data['high'].iloc[breakout_index] - data['low'].iloc[point1_idx_int]) / \
                     (data['high'].iloc[breakout_index] - data['low'].iloc[breakout_index])
        else:
            # 寻找回调高点
            point1_idx = data['high'].iloc[start_idx:end_idx].astype(float).idxmax()
            point1_idx_int = data.index.get_loc(point1_idx)
            retrace = (data['high'].iloc[point1_idx_int] - data['low'].iloc[breakout_index]) / \
                     (data['high'].iloc[breakout_index] - data['low'].iloc[breakout_index])
        
        if rules['retrace'][0] <= retrace <= rules['retrace'][1]:
            return {
                'index': point1_idx_int,
                'price': data['close'].iloc[point1_idx_int],
                'retrace': retrace
            }
        return None
    
    def _find_point2(self, data: pd.DataFrame, point1_index: int, seq_type: str) -> Optional[Dict]:
        """寻找点2"""
        rules = self.sequence_rules[seq_type]['points']['point2']
        start_idx = point1_index + rules['range'][0]
        end_idx = min(point1_index + rules['range'][1], len(data)-1)
        
        if seq_type == 'upward':
            # 寻找新高点
            point2_idx = data['high'].iloc[start_idx:end_idx].astype(float).idxmax()
            point2_idx_int = data.index.get_loc(point2_idx)
            extension = (data['high'].iloc[point2_idx_int] - data['low'].iloc[point1_index]) / \
                       (data['high'].iloc[point1_index] - data['low'].iloc[point1_index])
        else:
            # 寻找新低点
            point2_idx = data['low'].iloc[start_idx:end_idx].astype(float).idxmin()
            point2_idx_int = data.index.get_loc(point2_idx)
            extension = (data['high'].iloc[point1_index] - data['low'].iloc[point2_idx_int]) / \
                       (data['high'].iloc[point1_index] - data['low'].iloc[point1_index])
        
        if rules['extension'][0] <= extension <= rules['extension'][1]:
            return {
                'index': point2_idx_int,
                'price': data['close'].iloc[point2_idx_int],
                'extension': extension
            }
        return None
    
    def _find_point3(self, data: pd.DataFrame, point2_index: int, point1: Dict, seq_type: str) -> Optional[Dict]:
        """寻找点3"""
        rules = self.sequence_rules[seq_type]['points']['point3']
        start_idx = point2_index + rules['range'][0]
        end_idx = min(point2_index + rules['range'][1], len(data)-1)
        
        if seq_type == 'upward':
            # 寻找回调低点
            point3_idx = data['low'].iloc[start_idx:end_idx].astype(float).idxmin()
            point3_idx_int = data.index.get_loc(point3_idx)
            retrace = (data['high'].iloc[point2_index] - data['low'].iloc[point3_idx_int]) / \
                     (data['high'].iloc[point2_index] - data['low'].iloc[point1['index']])
        else:
            # 寻找回调高点
            point3_idx = data['high'].iloc[start_idx:end_idx].astype(float).idxmax()
            point3_idx_int = data.index.get_loc(point3_idx)
            retrace = (data['high'].iloc[point3_idx_int] - data['low'].iloc[point2_index]) / \
                     (data['high'].iloc[point1['index']] - data['low'].iloc[point2_index])
        
        if rules['retrace'][0] <= retrace <= rules['retrace'][1]:
            return {
                'index': point3_idx_int,
                'price': data['close'].iloc[point3_idx_int],
                'retrace': retrace
            }
        return None
    
    def _find_point4(self, data: pd.DataFrame, point3_index: int, point1: Dict, seq_type: str) -> Optional[Dict]:
        """寻找点4"""
        rules = self.sequence_rules[seq_type]['points']['point4']
        start_idx = point3_index + rules['range'][0]
        end_idx = min(point3_index + rules['range'][1], len(data)-1)
        
        if seq_type == 'upward':
            # 寻找支撑确认点
            point4_idx = data['low'].iloc[start_idx:end_idx].astype(float).idxmin()
            point4_idx_int = data.index.get_loc(point4_idx)
            valid = data['low'].iloc[point4_idx_int] >= data['low'].iloc[point1['index']]
        else:
            # 寻找阻力确认点
            point4_idx = data['high'].iloc[start_idx:end_idx].astype(float).idxmax()
            point4_idx_int = data.index.get_loc(point4_idx)
            valid = data['high'].iloc[point4_idx_int] <= data['high'].iloc[point1['index']]
        
        if valid:
            return {
                'index': point4_idx_int,
                'price': data['close'].iloc[point4_idx_int]
            }
        return None
    
    def _find_point5(self, data: pd.DataFrame, point4_index: int, point2: Dict, seq_type: str) -> Optional[Dict]:
        """寻找点5"""
        rules = self.sequence_rules[seq_type]['points']['point5']
        start_idx = point4_index + rules['range'][0]
        end_idx = min(point4_index + rules['range'][1], len(data)-1)
        
        if seq_type == 'upward':
            # 寻找目标点
            point5_idx = data['high'].iloc[start_idx:end_idx].astype(float).idxmax()
            point5_idx_int = data.index.get_loc(point5_idx)
            valid = data['high'].iloc[point5_idx_int] >= data['high'].iloc[point2['index']]
        else:
            # 寻找目标点
            point5_idx = data['low'].iloc[start_idx:end_idx].astype(float).idxmin()
            point5_idx_int = data.index.get_loc(point5_idx)
            valid = data['low'].iloc[point5_idx_int] <= data['low'].iloc[point2['index']]
        
        if valid:
            return {
                'index': point5_idx_int,
                'price': data['close'].iloc[point5_idx_int]
            }
        return None
    
    def _generate_sequence_sample(self, sequence: Dict, sample_id: int):
        """生成序列样本"""
        try:
            # 创建样本目录
            sample_dir = os.path.join(self.output_dir, f"sequence_{sample_id}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 生成图表
            chart_file = os.path.join(sample_dir, "chart.png")
            self._generate_chart(sequence, chart_file)
            
            # 保存序列信息
            info_file = os.path.join(sample_dir, "info.txt")
            self._save_sequence_info(sequence, info_file)
            
            logger.info(f"生成序列样本: {sample_id}")
            
        except Exception as e:
            logger.error(f"生成样本出错: {e}")
    
    def _generate_chart(self, sequence: Dict, filename: str):
        """生成序列图表"""
        data = sequence['data']
        start_idx = sequence['breakout_index'] - 20  # 包含突破前的走势
        end_idx = max(point['index'] for point in sequence['points'].values()) + 20
        
        # 截取数据
        chart_data = data.iloc[start_idx:end_idx]
        
        # 设置样式
        mc = mpf.make_marketcolors(up='lightgray', down='darkgray',
                                 edge='inherit',
                                 wick='inherit',
                                 volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # 添加均线
        apds = [
            mpf.make_addplot(chart_data['SMA20'], color='lightgray'),
            mpf.make_addplot(chart_data['SMA200'], color='darkgray')
        ]
        
        # 绘制图表
        mpf.plot(chart_data, type='candle',
                style=s,
                addplot=apds,
                volume=True,
                savefig=filename)
    
    def _save_sequence_info(self, sequence: Dict, filename: str):
        """保存序列信息"""
        with open(filename, 'w') as f:
            f.write("=== 序列信息 ===\n")
            f.write(f"类型: {sequence['type']}\n")
            f.write(f"突破时间: {sequence['data'].index[sequence['breakout_index']]}\n\n")
            
            f.write("=== 关键点位 ===\n")
            for point_name, point_data in sequence['points'].items():
                f.write(f"\n{point_name}:\n")
                f.write(f"时间: {sequence['data'].index[point_data['index']]}\n")
                f.write(f"价格: {point_data['price']}\n")
                if 'retrace' in point_data:
                    f.write(f"回调比例: {point_data['retrace']:.3f}\n")
                if 'extension' in point_data:
                    f.write(f"扩展比例: {point_data['extension']:.3f}\n")
    
    def _augment_sequence(self, sequence: Dict) -> List[Dict]:
        """数据增强"""
        augmented_sequences = []
        
        # 时间平移
        for shift in range(self.augmentation_config['time_shift'][0],
                         self.augmentation_config['time_shift'][1]+1, 2):
            aug_seq = self._time_shift_augmentation(sequence, shift)
            if aug_seq:
                augmented_sequences.append(aug_seq)
        
        # 添加噪声
        for _ in range(2):  # 生成2个带噪声的样本
            aug_seq = self._add_noise_augmentation(sequence)
            if aug_seq:
                augmented_sequences.append(aug_seq)
        
        # 成交量缩放
        for scale in [0.8, 1.2]:  # 生成2个成交量缩放的样本
            aug_seq = self._volume_scale_augmentation(sequence, scale)
            if aug_seq:
                augmented_sequences.append(aug_seq)
        
        return augmented_sequences
    
    def _time_shift_augmentation(self, sequence: Dict, shift: int) -> Optional[Dict]:
        """时间平移增强"""
        try:
            aug_seq = sequence.copy()
            aug_seq['breakout_index'] += shift
            
            # 调整所有点位
            aug_seq['points'] = {}
            for point_name, point_data in sequence['points'].items():
                aug_point = point_data.copy()
                aug_point['index'] += shift
                aug_seq['points'][point_name] = aug_point
            
            return aug_seq
            
        except Exception as e:
            logger.error(f"时间平移增强出错: {e}")
            return None
    
    def _add_noise_augmentation(self, sequence: Dict) -> Optional[Dict]:
        """添加噪声增强"""
        try:
            aug_seq = sequence.copy()
            data = sequence['data'].copy()
            
            # 添加随机噪声
            noise_level = np.random.uniform(*self.augmentation_config['noise_level'])
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, noise_level, len(data))
                data[col] *= (1 + noise)
            
            aug_seq['data'] = data
            return aug_seq
            
        except Exception as e:
            logger.error(f"添加噪声增强出错: {e}")
            return None
    
    def _volume_scale_augmentation(self, sequence: Dict, scale: float) -> Optional[Dict]:
        """成交量缩放增强"""
        try:
            aug_seq = sequence.copy()
            data = sequence['data'].copy()
            
            # 缩放成交量
            data['volume'] *= scale
            data['volume_ma20'] = data['volume'].rolling(window=20).mean()
            
            aug_seq['data'] = data
            return aug_seq
            
        except Exception as e:
            logger.error(f"成交量缩放增强出错: {e}")
            return None 
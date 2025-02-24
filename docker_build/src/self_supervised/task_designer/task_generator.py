import os
import json
from typing import Dict, List, Optional
import random
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger('task_generator')

class TaskGenerator:
    """自监督学习任务生成器"""
    
    def __init__(self):
        """初始化任务生成器"""
        self.tasks_dir = 'training_data/tasks/'
        self.sequences_dir = 'training_data/sequences/'
        
        # 确保目录存在
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        # 任务类型定义
        self.task_types = {
            'continuity': {
                'name': '序列连续性预测',
                'description': '预测序列的下一个关键点位',
                'difficulty': 1.0
            },
            'completeness': {
                'name': '模式完整性判断',
                'description': '判断序列是否完整有效',
                'difficulty': 0.8
            },
            'consistency': {
                'name': '时序一致性验证',
                'description': '验证序列在不同时间尺度下的一致性',
                'difficulty': 1.2
            }
        }
    
    def generate_tasks(self):
        """生成训练任务"""
        logger.info("开始生成训练任务")
        
        try:
            # 获取所有序列样本
            sequences = self._load_sequences()
            if not sequences:
                logger.error("未找到序列样本")
                return
            
            # 生成不同类型的任务
            tasks = []
            
            # 连续性预测任务
            continuity_tasks = self._generate_continuity_tasks(sequences)
            tasks.extend(continuity_tasks)
            
            # 完整性判断任务
            completeness_tasks = self._generate_completeness_tasks(sequences)
            tasks.extend(completeness_tasks)
            
            # 一致性验证任务
            consistency_tasks = self._generate_consistency_tasks(sequences)
            tasks.extend(consistency_tasks)
            
            # 保存任务
            self._save_tasks(tasks)
            
            logger.info(f"生成了 {len(tasks)} 个训练任务")
            
        except Exception as e:
            logger.error(f"生成任务出错: {e}")
    
    def _load_sequences(self) -> List[Dict]:
        """加载序列样本"""
        sequences = []
        
        try:
            # 遍历序列目录
            for seq_dir in os.listdir(self.sequences_dir):
                seq_path = os.path.join(self.sequences_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                
                # 读取序列信息
                info_file = os.path.join(seq_path, 'info.txt')
                if not os.path.exists(info_file):
                    continue
                
                # 解析序列信息
                sequence = self._parse_sequence_info(info_file)
                if sequence:
                    sequence['dir'] = seq_dir
                    sequence['chart_path'] = os.path.join(seq_path, 'chart.png')
                    sequences.append(sequence)
            
            logger.info(f"加载了 {len(sequences)} 个序列样本")
            return sequences
            
        except Exception as e:
            logger.error(f"加载序列样本出错: {e}")
            return []
    
    def _parse_sequence_info(self, info_file: str) -> Optional[Dict]:
        """解析序列信息文件"""
        try:
            sequence = {
                'type': None,
                'breakout_time': None,
                'points': {}
            }
            
            with open(info_file, 'r') as f:
                content = f.read()
                
                # 解析序列类型
                for line in content.split('\n'):
                    if '类型:' in line:
                        sequence['type'] = line.split(':')[1].strip()
                    elif '突破时间:' in line:
                        sequence['breakout_time'] = line.split(':')[1].strip()
                    elif 'point' in line and ':' in line:
                        point_name = line.strip(':')
                        point_info = {}
                        while True:
                            line = next(f, '').strip()
                            if not line or line.startswith('==='):
                                break
                            if ':' in line:
                                key, value = line.split(':')
                                point_info[key.strip()] = value.strip()
                        sequence['points'][point_name] = point_info
            
            return sequence
            
        except Exception as e:
            logger.error(f"解析序列信息出错: {e}")
            return None
    
    def _generate_continuity_tasks(self, sequences: List[Dict]) -> List[Dict]:
        """生成连续性预测任务"""
        tasks = []
        
        try:
            for seq in sequences:
                # 对每个序列生成多个预测任务
                points = list(seq['points'].keys())
                for i in range(len(points)-1):
                    task = {
                        'type': 'continuity',
                        'sequence_dir': seq['dir'],
                        'chart_path': seq['chart_path'],
                        'given_points': points[:i+1],
                        'target_point': points[i+1],
                        'sequence_type': seq['type'],
                        'difficulty': self.task_types['continuity']['difficulty']
                    }
                    tasks.append(task)
            
            logger.info(f"生成了 {len(tasks)} 个连续性预测任务")
            return tasks
            
        except Exception as e:
            logger.error(f"生成连续性预测任务出错: {e}")
            return []
    
    def _generate_completeness_tasks(self, sequences: List[Dict]) -> List[Dict]:
        """生成完整性判断任务"""
        tasks = []
        
        try:
            for seq in sequences:
                # 生成完整序列任务
                task = {
                    'type': 'completeness',
                    'sequence_dir': seq['dir'],
                    'chart_path': seq['chart_path'],
                    'points': list(seq['points'].keys()),
                    'sequence_type': seq['type'],
                    'is_complete': len(seq['points']) >= 4,  # 至少有4个点才算完整
                    'difficulty': self.task_types['completeness']['difficulty']
                }
                tasks.append(task)
                
                # 生成不完整序列任务（随机移除一些点）
                if len(seq['points']) >= 4:
                    incomplete_points = list(seq['points'].keys())
                    num_remove = random.randint(1, 2)
                    for _ in range(num_remove):
                        point_to_remove = random.choice(incomplete_points)
                        incomplete_points.remove(point_to_remove)
                    
                    task = {
                        'type': 'completeness',
                        'sequence_dir': seq['dir'],
                        'chart_path': seq['chart_path'],
                        'points': incomplete_points,
                        'sequence_type': seq['type'],
                        'is_complete': False,
                        'difficulty': self.task_types['completeness']['difficulty']
                    }
                    tasks.append(task)
            
            logger.info(f"生成了 {len(tasks)} 个完整性判断任务")
            return tasks
            
        except Exception as e:
            logger.error(f"生成完整性判断任务出错: {e}")
            return []
    
    def _generate_consistency_tasks(self, sequences: List[Dict]) -> List[Dict]:
        """生成一致性验证任务"""
        tasks = []
        
        try:
            # 按类型分组序列
            sequence_groups = {}
            for seq in sequences:
                seq_type = seq['type']
                if seq_type not in sequence_groups:
                    sequence_groups[seq_type] = []
                sequence_groups[seq_type].append(seq)
            
            # 为每个类型生成一致性任务
            for seq_type, group in sequence_groups.items():
                if len(group) >= 2:
                    # 随机选择序列对
                    for _ in range(min(len(group), 5)):  # 每个类型最多5个任务
                        seq1, seq2 = random.sample(group, 2)
                        task = {
                            'type': 'consistency',
                            'sequence1_dir': seq1['dir'],
                            'sequence2_dir': seq2['dir'],
                            'chart1_path': seq1['chart_path'],
                            'chart2_path': seq2['chart_path'],
                            'sequence_type': seq_type,
                            'points1': list(seq1['points'].keys()),
                            'points2': list(seq2['points'].keys()),
                            'difficulty': self.task_types['consistency']['difficulty']
                        }
                        tasks.append(task)
            
            logger.info(f"生成了 {len(tasks)} 个一致性验证任务")
            return tasks
            
        except Exception as e:
            logger.error(f"生成一致性验证任务出错: {e}")
            return []
    
    def _save_tasks(self, tasks: List[Dict]):
        """保存任务"""
        try:
            # 按任务类型分组保存
            for task_type in self.task_types:
                type_tasks = [t for t in tasks if t['type'] == task_type]
                if not type_tasks:
                    continue
                
                # 创建任务类型目录
                type_dir = os.path.join(self.tasks_dir, task_type)
                os.makedirs(type_dir, exist_ok=True)
                
                # 保存任务
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(type_dir, f"tasks_{timestamp}.json")
                
                with open(filename, 'w') as f:
                    json.dump({
                        'task_type': task_type,
                        'task_info': self.task_types[task_type],
                        'tasks': type_tasks
                    }, f, indent=2)
                
                logger.info(f"保存了 {len(type_tasks)} 个 {task_type} 任务到 {filename}")
            
        except Exception as e:
            logger.error(f"保存任务出错: {e}")
    
    def get_task_stats(self) -> Dict:
        """获取任务统计信息"""
        stats = {
            'total_tasks': 0,
            'by_type': {},
            'by_difficulty': {
                'easy': 0,    # difficulty < 0.8
                'medium': 0,  # 0.8 <= difficulty < 1.2
                'hard': 0     # difficulty >= 1.2
            }
        }
        
        try:
            # 遍历任务目录
            for task_type in self.task_types:
                type_dir = os.path.join(self.tasks_dir, task_type)
                if not os.path.exists(type_dir):
                    continue
                
                # 统计每个类型的任务数量
                type_count = 0
                for file in os.listdir(type_dir):
                    if not file.endswith('.json'):
                        continue
                    
                    with open(os.path.join(type_dir, file), 'r') as f:
                        task_data = json.load(f)
                        tasks = task_data.get('tasks', [])
                        type_count += len(tasks)
                        
                        # 统计难度分布
                        for task in tasks:
                            difficulty = task.get('difficulty', 1.0)
                            if difficulty < 0.8:
                                stats['by_difficulty']['easy'] += 1
                            elif difficulty < 1.2:
                                stats['by_difficulty']['medium'] += 1
                            else:
                                stats['by_difficulty']['hard'] += 1
                
                stats['by_type'][task_type] = type_count
                stats['total_tasks'] += type_count
            
            return stats
            
        except Exception as e:
            logger.error(f"获取任务统计信息出错: {e}")
            return stats 
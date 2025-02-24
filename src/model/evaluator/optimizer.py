"""优化器模块"""
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm
from scipy.optimize import minimize
import logging
import json
from datetime import datetime

class BaseOptimizer:
    """优化器基类"""
    
    def __init__(self, param_space: Dict[str, Any], objective_func: Callable):
        """初始化优化器
        
        Args:
            param_space: 参数空间
            objective_func: 目标函数
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        
    def save_optimization_history(self, filepath: str):
        """保存优化历史
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.optimization_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存优化历史失败: {str(e)}")

class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, n_jobs: int = -1) -> Tuple[Dict[str, Any], float]:
        """执行网格搜索优化
        
        Args:
            n_jobs: 并行作业数
            
        Returns:
            Tuple[Dict[str, Any], float]: 最优参数和对应的目标值
        """
        try:
            # 生成参数网格
            param_grid = ParameterGrid(self.param_space)
            
            best_score = float('-inf')
            best_params = None
            
            # 遍历所有参数组合
            for params in param_grid:
                score = self.objective_func(params)
                
                # 记录优化历史
                self.optimization_history.append({
                    'params': params,
                    'score': score,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # 更新最优结果
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            return best_params, best_score
            
        except Exception as e:
            self.logger.error(f"网格搜索优化失败: {str(e)}")
            return {}, float('-inf')

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def __init__(self,
                 param_space: Dict[str, Any],
                 objective_func: Callable,
                 n_initial_points: int = 5,
                 acquisition_func: str = 'ei',
                 exploration_weight: float = 0.1):
        """初始化贝叶斯优化器
        
        Args:
            param_space: 参数空间
            objective_func: 目标函数
            n_initial_points: 初始随机点数量
            acquisition_func: 采集函数类型 ('ei', 'pi', 'ucb')
            exploration_weight: 探索权重
        """
        super().__init__(param_space, objective_func)
        self.n_initial_points = n_initial_points
        self.acquisition_func = acquisition_func
        self.exploration_weight = exploration_weight
        self.X = []  # 已评估的参数点
        self.y = []  # 对应的目标值
        
    def optimize(self, n_iterations: int = 50) -> Tuple[Dict[str, Any], float]:
        """执行贝叶斯优化
        
        Args:
            n_iterations: 优化迭代次数
            
        Returns:
            Tuple[Dict[str, Any], float]: 最优参数和对应的目标值
        """
        try:
            # 初始随机采样
            self._initial_sampling()
            
            # 主优化循环
            for _ in range(n_iterations):
                # 选择下一个评估点
                next_params = self._select_next_point()
                
                # 评估目标函数
                score = self.objective_func(next_params)
                
                # 更新观测数据
                self.X.append(list(next_params.values()))
                self.y.append(score)
                
                # 记录优化历史
                self.optimization_history.append({
                    'params': next_params,
                    'score': score,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            # 返回最优结果
            best_idx = np.argmax(self.y)
            best_params = dict(zip(self.param_space.keys(), self.X[best_idx]))
            return best_params, self.y[best_idx]
            
        except Exception as e:
            self.logger.error(f"贝叶斯优化失败: {str(e)}")
            return {}, float('-inf')
            
    def _initial_sampling(self):
        """初始随机采样"""
        for _ in range(self.n_initial_points):
            # 随机生成参数
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, (list, tuple)):
                    params[param_name] = np.random.choice(param_range)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            
            # 评估目标函数
            score = self.objective_func(params)
            
            # 记录结果
            self.X.append(list(params.values()))
            self.y.append(score)
            
            # 记录优化历史
            self.optimization_history.append({
                'params': params,
                'score': score,
                'timestamp': datetime.utcnow().isoformat()
            })
            
    def _select_next_point(self) -> Dict[str, Any]:
        """选择下一个评估点
        
        Returns:
            Dict[str, Any]: 下一组参数
        """
        X = np.array(self.X)
        y = np.array(self.y)
        
        # 计算高斯过程回归
        mean = np.mean(y)
        std = np.std(y) if len(y) > 1 else 1.0
        
        def acquisition(x):
            # 根据选择的采集函数计算采集值
            if self.acquisition_func == 'ei':  # Expected Improvement
                imp = (x - np.max(y)) / (std + 1e-9)
                return -(mean + std * (imp * norm.cdf(imp) + norm.pdf(imp)))
            elif self.acquisition_func == 'pi':  # Probability of Improvement
                return -norm.cdf((x - np.max(y)) / (std + 1e-9))
            else:  # Upper Confidence Bound
                return -(x + self.exploration_weight * std)
        
        # 优化采集函数
        best_acquisition = float('inf')
        best_params = None
        
        # 在参数空间中采样多个点
        n_samples = 1000
        for _ in range(n_samples):
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, (list, tuple)):
                    params[param_name] = np.random.choice(param_range)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            
            acq_value = acquisition(self.objective_func(params))
            
            if acq_value < best_acquisition:
                best_acquisition = acq_value
                best_params = params
        
        return best_params 
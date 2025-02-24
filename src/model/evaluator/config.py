"""评估和优化配置模块"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import yaml
import logging
from pathlib import Path

@dataclass
class OptimizerConfig:
    """优化器配置"""
    
    # 优化方法
    method: str = 'bayesian'  # 'bayesian' 或 'grid'
    
    # 贝叶斯优化配置
    n_initial_points: int = 5
    n_iterations: int = 50
    acquisition_func: str = 'ei'  # 'ei', 'pi', 或 'ucb'
    exploration_weight: float = 0.1
    
    # 网格搜索配置
    n_jobs: int = -1
    
    # 参数空间
    param_space: Dict[str, Any] = field(default_factory=dict)
    
    # 优化目标
    optimization_metric: str = 'f1'  # 优化目标指标
    optimization_direction: str = 'maximize'  # 'maximize' 或 'minimize'
    
    # 早停配置
    early_stopping_rounds: int = 10
    early_stopping_threshold: float = 0.001

@dataclass
class EvaluationConfig:
    """评估配置"""
    
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor'
    ])
    
    # 交易评估配置
    risk_free_rate: float = 0.0
    initial_capital: float = 10000.0
    position_size: float = 0.1
    
    # 验证配置
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    # 稳定性评估配置
    n_stability_runs: int = 5
    stability_threshold: float = 0.1

@dataclass
class Config:
    """总配置"""
    
    # 基本配置
    experiment_name: str
    model_name: str
    strategy_name: str
    symbol: str
    timeframe: str
    
    # 评估和优化配置
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # 输出配置
    output_dir: str = 'outputs'
    save_history: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """从YAML文件加载配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            Config: 配置实例
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 创建评估配置
            evaluation_config = EvaluationConfig(
                **config_dict.get('evaluation', {})
            )
            
            # 创建优化器配置
            optimizer_config = OptimizerConfig(
                **config_dict.get('optimizer', {})
            )
            
            # 创建总配置
            return cls(
                experiment_name=config_dict['experiment_name'],
                model_name=config_dict['model_name'],
                strategy_name=config_dict['strategy_name'],
                symbol=config_dict['symbol'],
                timeframe=config_dict['timeframe'],
                evaluation=evaluation_config,
                optimizer=optimizer_config,
                output_dir=config_dict.get('output_dir', 'outputs'),
                save_history=config_dict.get('save_history', True),
                save_checkpoints=config_dict.get('save_checkpoints', True),
                checkpoint_frequency=config_dict.get('checkpoint_frequency', 10)
            )
            
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            raise
            
    def save(self, filepath: str):
        """保存配置到YAML文件
        
        Args:
            filepath: 配置文件路径
        """
        try:
            # 确保目录存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典
            config_dict = {
                'experiment_name': self.experiment_name,
                'model_name': self.model_name,
                'strategy_name': self.strategy_name,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'evaluation': {
                    'metrics': self.evaluation.metrics,
                    'risk_free_rate': self.evaluation.risk_free_rate,
                    'initial_capital': self.evaluation.initial_capital,
                    'position_size': self.evaluation.position_size,
                    'validation_split': self.evaluation.validation_split,
                    'test_split': self.evaluation.test_split,
                    'cross_validation_folds': self.evaluation.cross_validation_folds,
                    'n_stability_runs': self.evaluation.n_stability_runs,
                    'stability_threshold': self.evaluation.stability_threshold
                },
                'optimizer': {
                    'method': self.optimizer.method,
                    'n_initial_points': self.optimizer.n_initial_points,
                    'n_iterations': self.optimizer.n_iterations,
                    'acquisition_func': self.optimizer.acquisition_func,
                    'exploration_weight': self.optimizer.exploration_weight,
                    'n_jobs': self.optimizer.n_jobs,
                    'param_space': self.optimizer.param_space,
                    'optimization_metric': self.optimizer.optimization_metric,
                    'optimization_direction': self.optimizer.optimization_direction,
                    'early_stopping_rounds': self.optimizer.early_stopping_rounds,
                    'early_stopping_threshold': self.optimizer.early_stopping_threshold
                },
                'output_dir': self.output_dir,
                'save_history': self.save_history,
                'save_checkpoints': self.save_checkpoints,
                'checkpoint_frequency': self.checkpoint_frequency
            }
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
                
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
            raise 
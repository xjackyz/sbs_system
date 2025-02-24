"""
训练配置文件
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练基本配置
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 优化器配置
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    warmup_steps: int = 1000
    
    # 训练策略
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    
    # 模型保存
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 5
    
    # 混合精度训练
    use_amp: bool = True
    fp16_opt_level: str = "O1"
    
    # 分布式训练
    distributed_training: bool = False
    world_size: int = 1
    local_rank: int = 0

# 默认配置
DEFAULT_CONFIG = TrainingConfig()

def load_config() -> TrainingConfig:
    """加载训练配置"""
    return DEFAULT_CONFIG 
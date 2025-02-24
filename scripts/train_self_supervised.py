import os
import sys
import argparse
from datetime import datetime
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_supervised.trainer.self_supervised_trainer import SelfSupervisedTrainer
from src.self_supervised.model.sequence_model import SequenceModel, ModelConfig
from src.utils.logger import setup_logger
from config.config import SELF_SUPERVISED_CONFIG, TRAINING_WORKFLOW

logger = setup_logger('self_supervised_training')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行自监督学习训练')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--model_dir', type=str, default='models/self_supervised',
                       help='模型保存目录')
    parser.add_argument('--stage', type=int, default=1,
                       help='训练阶段(1-3)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--evaluate', action='store_true',
                       help='是否只进行评估')
    
    return parser.parse_args()

def train_stage(trainer: SelfSupervisedTrainer, stage: int, num_epochs: int):
    """执行特定阶段的训练"""
    stage_info = TRAINING_WORKFLOW['training_stages'][f'stage{stage}']
    logger.info(f"开始{stage_info['name']}阶段训练")
    logger.info(f"训练目标: {stage_info['objective']}")
    
    # 设置模型训练阶段
    if stage == 1:
        trainer.model.train_stage1()
    elif stage == 2:
        trainer.model.train_stage2()
    else:
        trainer.model.train_stage3()
    
    # 训练指定轮数
    trainer.train(num_epochs=num_epochs)
    
    # 阶段性评估
    metrics = trainer.evaluate()
    logger.info(f"阶段{stage}训练完成，评估结果:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    return metrics

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 创建保存目录
        os.makedirs(args.model_dir, exist_ok=True)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 创建模型配置
        model_config = ModelConfig(
            image_size=SELF_SUPERVISED_CONFIG['model']['image_size'],
            sequence_length=SELF_SUPERVISED_CONFIG['model']['sequence_length'],
            hidden_size=SELF_SUPERVISED_CONFIG['model']['hidden_size'],
            num_heads=SELF_SUPERVISED_CONFIG['model']['num_heads'],
            num_layers=SELF_SUPERVISED_CONFIG['model']['num_layers']
        )
        
        # 创建模型
        model = SequenceModel(config=model_config)
        
        # 创建训练器
        trainer = SelfSupervisedTrainer(
            model=model,
            data_dir=args.data_dir,
            save_dir=args.model_dir,
            device=device
        )
        
        # 如果指定了检查点，加载它
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"从检查点恢复: {args.resume}")
        
        # 如果是评估模式
        if args.evaluate:
            if not args.resume:
                raise ValueError("评估模式需要指定检查点路径")
            
            logger.info("开始评估...")
            metrics = trainer.evaluate()
            
            logger.info("评估结果:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
                
        # 否则开始训练
        else:
            # 获取当前阶段的训练轮数
            stage_info = TRAINING_WORKFLOW['training_stages'][f'stage{args.stage}']
            num_epochs = int(stage_info['duration'].split()[0])
            
            # 执行当前阶段的训练
            train_stage(trainer, args.stage, num_epochs)
            
            # 如果是最后一个阶段，进行完整评估
            if args.stage == 3:
                logger.info("所有阶段训练完成，开始最终评估...")
                metrics = trainer.evaluate()
                
                logger.info("最终评估结果:")
                for metric_name, value in metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
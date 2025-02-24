import os
import sys
import argparse
from datetime import datetime
import torch
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_supervised.trainer.self_supervised_trainer import SelfSupervisedTrainer
from src.self_supervised.model.sequence_model import SequenceModel, ModelConfig
from src.self_supervised.utils.output_formatter import OutputFormatter, OutputRequirements
from src.utils.logger import setup_logger
from config.config import SELF_SUPERVISED_CONFIG

logger = setup_logger('model_evaluation')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估模型性能')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='评估数据目录')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='评估结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--validity_threshold', type=float, default=0.8,
                       help='序列有效性阈值')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='置信度阈值')
    
    return parser.parse_args()

def evaluate_model(args):
    """评估模型"""
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 创建输出格式化器
        output_requirements = OutputRequirements(
            sequence_validity_threshold=args.validity_threshold,
            confidence_threshold=args.confidence_threshold
        )
        formatter = OutputFormatter(requirements=output_requirements)
        
        # 加载模型配置
        checkpoint = torch.load(args.model_path, map_location=device)
        model_config = ModelConfig(**checkpoint['config'])
        
        # 创建模型
        model = SequenceModel(config=model_config)
        
        # 创建训练器
        trainer = SelfSupervisedTrainer(
            model=model,
            data_dir=args.data_dir,
            save_dir=args.output_dir,
            device=device
        )
        
        # 加载模型权重
        trainer.load_checkpoint(args.model_path)
        
        # 进行评估
        logger.info("开始评估...")
        raw_metrics = trainer.evaluate()
        
        # 格式化评估结果
        formatted_metrics = formatter.format_evaluation_report(raw_metrics)
        
        # 保存评估结果
        results_file = os.path.join(
            args.output_dir,
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(formatted_metrics, f, indent=2)
            
        logger.info(f"评估结果已保存到: {results_file}")
        
        # 输出主要指标
        logger.info("\n=== 评估结果 ===")
        
        # 性能指标
        metrics = formatted_metrics['performance_metrics']
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"精确率: {metrics['precision']:.4f}")
        logger.info(f"召回率: {metrics['recall']:.4f}")
        
        # 错误分析
        error_analysis = formatted_metrics['error_analysis']
        logger.info(f"\n错误分析:")
        logger.info(f"假阳性数量: {error_analysis['false_positives']}")
        logger.info(f"假阴性数量: {error_analysis['false_negatives']}")
        
        # 序列验证结果
        if 'validation_results' in raw_metrics:
            val_results = raw_metrics['validation_results']
            valid_sequences = sum(1 for r in val_results if r.is_valid)
            total_sequences = len(val_results)
            avg_score = sum(r.score for r in val_results) / total_sequences
            
            logger.info(f"\n序列验证:")
            logger.info(f"有效序列比例: {valid_sequences}/{total_sequences} "
                       f"({valid_sequences/total_sequences*100:.2f}%)")
            logger.info(f"平均验证分数: {avg_score:.4f}")
            
            # 按阶段统计
            stage_stats = {}
            for result in val_results:
                if result.stage not in stage_stats:
                    stage_stats[result.stage] = {
                        'count': 0,
                        'valid': 0,
                        'total_score': 0
                    }
                stage_stats[result.stage]['count'] += 1
                if result.is_valid:
                    stage_stats[result.stage]['valid'] += 1
                stage_stats[result.stage]['total_score'] += result.score
            
            logger.info("\n各阶段统计:")
            for stage, stats in stage_stats.items():
                avg_score = stats['total_score'] / stats['count']
                valid_ratio = stats['valid'] / stats['count'] * 100
                logger.info(f"{stage}阶段:")
                logger.info(f"  - 数量: {stats['count']}")
                logger.info(f"  - 有效率: {valid_ratio:.2f}%")
                logger.info(f"  - 平均分数: {avg_score:.4f}")
        
    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        sys.exit(1)

def main():
    """主函数"""
    args = parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main() 
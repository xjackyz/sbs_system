import argparse
import logging
import yaml
from pathlib import Path
import torch
from datetime import datetime
import cv2
import numpy as np
from typing import Optional, Dict, Any

from src.model.sbs_analyzer import SBSAnalyzer
from src.training.train_sbs_analyzer import SBSTrainer
from src.utils.trading_alerts import TradingAlertManager
from src.data.dataset import SBSDataset

class SBSSystem:
    def __init__(self, config_path: str):
        """初始化SBS系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 设置日志
        self._setup_logging()
        
        # 初始化设备
        self.device = torch.device(self.config['system']['device'])
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化交易提醒管理器
        self.alert_manager = TradingAlertManager(
            config_path=self.config['alerts']['config_path']
        )
        
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['system']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'sbs_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_model(self) -> SBSAnalyzer:
        """初始化模型"""
        model = SBSAnalyzer(
            base_model=self.config['model']['base_model'],
            pretrained=True,
            device=self.device
        )
        
        # 加载训练好的权重
        if 'weights_path' in self.config['model']:
            weights_path = self.config['model']['weights_path']
            if Path(weights_path).exists():
                self.logger.info(f"加载模型权重: {weights_path}")
                model = SBSAnalyzer.load_model(weights_path, self.device)
            else:
                self.logger.warning(f"模型权重文件不存在: {weights_path}")
                
        return model
    
    def process_image(self, 
                     image_path: str,
                     symbol: str = 'unknown',
                     timeframe: str = '1h') -> Dict[str, Any]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            symbol: 交易对
            timeframe: 时间周期
        """
        # 读取并预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1) / 255.0
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std
        
        # 预测
        step, confidence = self.model.predict_step(image.to(self.device))
        
        # 生成提醒
        if confidence >= self.config['alerts']['confidence_threshold']:
            alert = self.alert_manager.generate_alert(
                step=step,
                confidence=confidence,
                symbol=symbol,
                timeframe=timeframe
            )
        else:
            alert = None
            self.logger.info(f"置信度 {confidence:.2f} 低于阈值，不生成提醒")
            
        return {
            'step': step,
            'confidence': confidence,
            'alert': alert
        }
    
    def train_model(self, training_config_path: str):
        """训练模型
        
        Args:
            training_config_path: 训练配置文件路径
        """
        trainer = SBSTrainer(training_config_path)
        trainer.train()
        
    def analyze_historical_data(self,
                              data_dir: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None):
        """分析历史数据
        
        Args:
            data_dir: 数据目录
            start_time: 开始时间
            end_time: 结束时间
        """
        # 加载数据集
        dataset = SBSDataset(
            data_dir=data_dir,
            transform=None  # 使用默认转换
        )
        
        results = []
        for image, _ in dataset:
            # 预测
            step, confidence = self.model.predict_step(image.to(self.device))
            
            results.append({
                'step': step,
                'confidence': confidence
            })
            
        # 分析结果
        step_counts = {}
        total_confidence = 0
        
        for result in results:
            step = result['step']
            step_counts[step.name] = step_counts.get(step.name, 0) + 1
            total_confidence += result['confidence']
            
        analysis = {
            'total_samples': len(results),
            'step_distribution': step_counts,
            'average_confidence': total_confidence / len(results) if results else 0
        }
        
        self.logger.info("历史数据分析结果:")
        self.logger.info(f"总样本数: {analysis['total_samples']}")
        self.logger.info(f"步骤分布: {analysis['step_distribution']}")
        self.logger.info(f"平均置信度: {analysis['average_confidence']:.2f}")
        
        return analysis
    
    def run_real_time_monitoring(self, input_source: str):
        """运行实时监控
        
        Args:
            input_source: 输入源（可以是视频流URL或截图目录）
        """
        self.logger.info(f"开始实时监控: {input_source}")
        
        # TODO: 实现实时监控逻辑
        
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'alerts_count': len(self.alert_manager.alerts_history)
        }

def main():
    parser = argparse.ArgumentParser(description='SBS交易系统')
    parser.add_argument('--config', type=str, required=True,
                       help='系统配置文件路径')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'analyze', 'monitor'],
                       help='运行模式')
    parser.add_argument('--input', type=str,
                       help='输入文件或目录')
    args = parser.parse_args()
    
    # 初始化系统
    system = SBSSystem(args.config)
    
    # 根据模式运行
    if args.mode == 'train':
        system.train_model(args.input)
    elif args.mode == 'predict':
        result = system.process_image(args.input)
        print(f"预测结果: {result}")
    elif args.mode == 'analyze':
        analysis = system.analyze_historical_data(args.input)
        print(f"分析结果: {analysis}")
    elif args.mode == 'monitor':
        system.run_real_time_monitoring(args.input)
        
if __name__ == '__main__':
    main() 
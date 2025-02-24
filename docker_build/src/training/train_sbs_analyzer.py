import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Dict, Any
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

from src.model.sbs_analyzer import SBSAnalyzer, SBSSteps
from src.data.dataset import SBSDataset

class SBSTrainer:
    def __init__(self, config_path: str):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 设置设备
        self.device = torch.device(self.config['training']['device'])
        
        # 设置随机种子
        torch.manual_seed(self.config['training']['seed'])
        
        # 初始化模型
        self.model = SBSAnalyzer(
            base_model=self.config['model']['base_model'],
            pretrained=True,
            freeze_backbone=True,
            device=self.device
        )
        
        # 设置数据转换
        self.transform = transforms.Compose([
            transforms.Resize(self.config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化数据集和加载器
        self._init_dataloaders()
        
        # 设置损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        # 设置日志
        self._setup_logging()
        
    def _init_dataloaders(self):
        """初始化数据加载器"""
        # 训练集
        train_dataset = SBSDataset(
            data_dir=self.config['data']['train_dir'],
            transform=self.transform
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )
        
        # 验证集
        val_dataset = SBSDataset(
            data_dir=self.config['data']['val_dir'],
            transform=self.transform
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
        
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """初始化优化器"""
        params = self.model.get_training_parameters(
            learning_rate=self.config['training']['learning_rate']
        )
        return optim.Adam(params)
    
    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """初始化学习率调度器"""
        if self.config['training'].get('use_scheduler', False):
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=3,
                verbose=True
            )
        return None
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': 100. * correct / total
        }
    
    def train(self):
        """训练模型"""
        best_acc = 0
        epochs_without_improvement = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # 训练阶段
            train_metrics = self.train_epoch()
            self.logger.info(f"Training - Loss: {train_metrics['loss']:.4f}, "
                           f"Accuracy: {train_metrics['accuracy']:.2f}%")
            
            # 验证阶段
            val_metrics = self.validate()
            self.logger.info(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                           f"Accuracy: {val_metrics['val_accuracy']:.2f}%")
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_accuracy'])
            
            # 保存最佳模型
            if val_metrics['val_accuracy'] > best_acc:
                best_acc = val_metrics['val_accuracy']
                epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
            else:
                epochs_without_improvement += 1
            
            # 提前停止
            if epochs_without_improvement >= self.config['training']['early_stopping']:
                self.logger.info("Early stopping triggered")
                break
            
            # 解冻更多层
            if epoch == self.config['training'].get('unfreeze_epoch', float('inf')):
                self.logger.info("Unfreezing more layers for fine-tuning")
                self.model.unfreeze_layers(self.config['training']['unfreeze_layers'])
                
        self.logger.info(f"Best validation accuracy: {best_acc:.2f}%")
        
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        save_dir = Path(self.config['training']['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        torch.save(checkpoint, save_dir / filename)
        self.logger.info(f"Checkpoint saved to {save_dir / filename}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='训练配置文件路径')
    args = parser.parse_args()
    
    trainer = SBSTrainer(args.config)
    trainer.train() 
import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.sbs_analyzer import SBSAnalyzer, SBSSteps
from src.training.train_sbs_analyzer import SBSTrainer
from src.data.dataset import SBSDataset

class TestSBSSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # 设置设备
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.logger.info(f"使用设备: {cls.device}")
        
        # 创建测试目录
        cls.test_dir = Path('test_data')
        cls.test_dir.mkdir(exist_ok=True)
        
        # 初始化模型
        cls.model = SBSAnalyzer(
            base_model='resnet50',
            pretrained=True,
            device=cls.device
        )
        
    def test_1_model_initialization(self):
        """测试模型初始化"""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(SBSSteps), 5)  # 确认有5个交易步骤
        
    def test_2_model_forward(self):
        """测试模型前向传播"""
        # 创建模拟输入
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # 测试前向传播
        with torch.no_grad():
            output = self.model(dummy_input)
        
        self.assertEqual(output.shape, (1, len(SBSSteps)))
        
    def test_3_prediction(self):
        """测试预测功能"""
        # 创建模拟图像
        test_image = torch.randn(3, 224, 224).to(self.device)
        
        # 进行预测
        step, confidence = self.model.predict_step(test_image)
        
        self.assertIsInstance(step, SBSSteps)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
        
    def test_4_save_load_model(self):
        """测试模型保存和加载"""
        # 保存模型
        save_path = self.test_dir / 'test_model.pth'
        self.model.save_model(str(save_path))
        
        # 加载模型
        loaded_model = SBSAnalyzer.load_model(str(save_path), self.device)
        
        # 验证加载的模型
        self.assertEqual(
            loaded_model.base_model_name,
            self.model.base_model_name
        )
        
    def test_5_training_setup(self):
        """测试训练设置"""
        config_path = 'config/training_config.yaml'
        
        try:
            trainer = SBSTrainer(config_path)
            self.assertIsNotNone(trainer)
            self.assertIsNotNone(trainer.optimizer)
            self.assertIsNotNone(trainer.criterion)
        except Exception as e:
            self.fail(f"训练器初始化失败: {e}")
            
    def test_6_data_pipeline(self):
        """测试数据管道"""
        # 创建测试数据集
        test_dataset = SBSDataset(
            data_dir=str(self.test_dir),
            transform=None
        )
        
        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False
        )
        
        self.assertIsNotNone(test_loader)
        
    def test_7_model_optimization(self):
        """测试模型优化功能"""
        # 测试层解冻
        initial_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        self.model.unfreeze_layers(2)
        after_unfreeze = sum(1 for p in self.model.parameters() if not p.requires_grad)
        
        self.assertLess(after_unfreeze, initial_frozen)
        
    def test_8_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建模拟交易图表
        test_chart = torch.randn(3, 224, 224).to(self.device)
        
        # 1. 预测步骤
        step, confidence = self.model.predict_step(test_chart)
        self.logger.info(f"预测步骤: {step.name}, 置信度: {confidence:.2f}")
        
        # 2. 生成交易提醒
        if confidence > 0.8:  # 置信度阈值
            self.logger.info(f"交易提醒: 检测到{step.name}模式")
            
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试文件
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            
if __name__ == '__main__':
    unittest.main() 
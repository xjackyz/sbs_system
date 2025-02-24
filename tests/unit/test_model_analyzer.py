import unittest
import torch
import numpy as np
import os
from PIL import Image
from src.model.llava_analyzer import LLaVAAnalyzer
from src.model.model_optimizer import ModelOptimizer
from config.config import PATHS

class TestModelAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类的设置，只运行一次"""
        # 创建测试目录
        cls.test_dir = "tests/test_data"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # 创建模拟的模型文件
        cls.test_model_path = os.path.join(cls.test_dir, "test_model.pt")
        
        # 创建一个简单的测试模型并保存
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 56 * 56, 10)
        )
        torch.save(model, cls.test_model_path)
        
        # 初始化分析器和优化器
        cls.analyzer = LLaVAAnalyzer()
        cls.optimizer = ModelOptimizer(model_path=cls.test_model_path, device='cpu')
        
    def setUp(self):
        """每个测试方法前的设置"""
        # 创建测试图像文件
        self.test_image_path = os.path.join(self.test_dir, "test_chart.png")
        
        # 创建一个简单的测试图像
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image.save(self.test_image_path)
        
        # 创建测试张量
        self.test_tensor = torch.randn(1, 3, 224, 224)
        
    async def test_analyze_chart(self):
        """测试图表分析"""
        result = await self.analyzer.analyze_chart(self.test_image_path)
        
        self.assertIsNotNone(result)
        self.assertIn('trade_signal', result)
        self.assertIn('market_analysis', result)
        self.assertIn('confidence', result)
        
    def test_process_image(self):
        """测试图像处理"""
        result = self.analyzer._process_image(self.test_image_path)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('preprocessed_image', result)
        
    def test_parse_response(self):
        """测试响应解析"""
        test_response = """
        市场分析:
        趋势: 上升
        支撑位: 15000
        阻力位: 15500
        成交量: 较高
        
        交易信号:
        方向: 买入
        入场价: 15200
        止损: 15000
        目标价: 15500
        置信度: 0.85
        """
        
        result = self.analyzer._parse_response(test_response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('market_analysis', result)
        self.assertIn('trade_signal', result)
        
    def test_extract_sequence_points(self):
        """测试序列点提取"""
        test_response = """
        序列点位:
        Point1: 15000 (2024-01-01 10:00)
        Point2: 15200 (2024-01-01 10:30)
        Point3: 15100 (2024-01-01 11:00)
        Point4: 15300 (2024-01-01 11:30)
        Point5: 15250 (2024-01-01 12:00)
        """
        
        result = self.analyzer._extract_sequence_points(test_response)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)
        self.assertIn('point1', result)
        
    def test_validate_sequence(self):
        """测试序列验证"""
        test_sequence = {
            'point1': {'price': 15000, 'time': '10:00'},
            'point2': {'price': 15200, 'time': '10:30'},
            'point3': {'price': 15100, 'time': '11:00'},
            'point4': {'price': 15300, 'time': '11:30'},
            'point5': {'price': 15250, 'time': '12:00'}
        }
        
        result = self.analyzer._validate_sequence(str(test_sequence))
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)
        self.assertIn('confidence', result)
        
    def test_generate_trade_signal(self):
        """测试交易信号生成"""
        test_response = """
        交易信号:
        类型: 买入
        入场价: 15200
        止损价: 15000
        目标价: 15500
        置信度: 0.85
        """
        
        result = self.analyzer._generate_trade_signal(test_response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('type', result)
        self.assertIn('entry_price', result)
        self.assertIn('stop_loss', result)
        self.assertIn('take_profit', result)
        
    def test_extract_price_level(self):
        """测试价格水平提取"""
        test_response = "支撑位: 15000"
        level = self.analyzer._extract_price_level(test_response, "支撑位")
        
        self.assertIsInstance(level, float)
        self.assertEqual(level, 15000.0)
        
    def test_analyze_market_conditions(self):
        """测试市场条件分析"""
        test_response = """
        市场分析:
        趋势: 上升
        波动性: 中等
        成交量: 高
        市场结构: 看涨
        """
        
        result = self.analyzer._analyze_market_conditions(test_response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('trend', result)
        self.assertIn('volatility', result)
        self.assertIn('volume', result)
        
    def test_model_optimization(self):
        """测试模型优化"""
        self.optimizer.optimize_model(batch_size=1)
        
        # 测试推理
        with torch.no_grad():
            result = self.optimizer.inference(self.test_tensor)
            
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效图像路径
        with self.assertRaises(FileNotFoundError):
            self.analyzer._process_image("invalid_path.png")
            
        # 测试无效响应
        with self.assertRaises(ValueError):
            self.analyzer._parse_response("")
            
    def tearDown(self):
        """每个测试方法后的清理"""
        # 删除测试图像
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        
    @classmethod
    def tearDownClass(cls):
        """测试类的清理，只运行一次"""
        # 删除测试目录
        if os.path.exists("tests/test_data"):
            import shutil
            shutil.rmtree("tests/test_data")
            
        cls.analyzer.cleanup()
        cls.optimizer.cleanup()

if __name__ == '__main__':
    unittest.main() 
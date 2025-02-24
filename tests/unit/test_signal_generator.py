import unittest
import numpy as np
from datetime import datetime
from src.signal.signal_generator import SignalGenerator, SignalConfig

class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = SignalConfig(
            min_confidence=0.75,
            max_risk_ratio=0.02,
            min_reward_ratio=2.0,
            entry_zone_size=0.001,
            max_stop_distance=0.02,
            min_volume=1000
        )
        self.generator = SignalGenerator(self.config)
        
        # 创建测试分析结果
        self.test_analysis = {
            'market_analysis': {
                'trend': 'upward',
                'strength': 0.8,
                'volume': 'high',
                'support_levels': [15000],
                'resistance_levels': [15500]
            },
            'sequence_analysis': {
                'type': 'sbs_upward',
                'confidence': 0.85,
                'points': {
                    'point1': {'price': 15000, 'time': '10:00'},
                    'point2': {'price': 15200, 'time': '10:30'},
                    'point3': {'price': 15100, 'time': '11:00'},
                    'point4': {'price': 15300, 'time': '11:30'},
                    'point5': {'price': 15250, 'time': '12:00'}
                }
            }
        }
        
    def test_generate_signal(self):
        """测试信号生成"""
        signal = self.generator.generate_signal(self.test_analysis)
        
        self.assertIsNotNone(signal)
        self.assertIn('direction', signal)
        self.assertIn('entry', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('take_profit', signal)
        self.assertIn('confidence', signal)
        
    def test_validate_analysis(self):
        """测试分析结果验证"""
        is_valid = self.generator.validate_analysis(self.test_analysis)
        self.assertTrue(is_valid)
        
        # 测试无效分析结果
        invalid_analysis = {
            'market_analysis': {},
            'sequence_analysis': {}
        }
        is_valid = self.generator.validate_analysis(invalid_analysis)
        self.assertFalse(is_valid)
        
    def test_calculate_entry(self):
        """测试入场点计算"""
        entry = self.generator.calculate_entry(self.test_analysis)
        
        self.assertIsInstance(entry, dict)
        self.assertIn('price', entry)
        self.assertIn('zone', entry)
        
    def test_calculate_stop_loss(self):
        """测试止损计算"""
        stop_loss = self.generator.calculate_stop_loss(self.test_analysis)
        
        self.assertIsInstance(stop_loss, float)
        self.assertLess(stop_loss, float(self.test_analysis['sequence_analysis']['points']['point1']['price']))
        
    def test_calculate_take_profit(self):
        """测试获利目标计算"""
        take_profit = self.generator.calculate_take_profit(self.test_analysis)
        
        self.assertIsInstance(take_profit, list)
        self.assertGreater(len(take_profit), 0)
        self.assertGreater(take_profit[0], float(self.test_analysis['sequence_analysis']['points']['point5']['price']))
        
    def test_calculate_confidence(self):
        """测试置信度计算"""
        confidence = self.generator.calculate_confidence(self.test_analysis)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_determine_direction(self):
        """测试方向判断"""
        direction = self.generator._determine_direction(self.test_analysis)
        
        self.assertIn(direction, ['long', 'short', None])
        
    def test_validate_basic_requirements(self):
        """测试基本要求验证"""
        is_valid = self.generator._validate_basic_requirements(self.test_analysis)
        self.assertTrue(is_valid)
        
    def test_validate_sequence(self):
        """测试序列验证"""
        is_valid = self.generator._validate_sequence(self.test_analysis)
        self.assertTrue(is_valid)
        
    def test_validate_market_conditions(self):
        """测试市场条件验证"""
        is_valid = self.generator._validate_market_conditions(self.test_analysis)
        self.assertTrue(is_valid)
        
    def test_validate_pattern(self):
        """测试模式验证"""
        is_valid = self.generator._validate_pattern(self.test_analysis)
        self.assertTrue(is_valid)
        
    def test_validate_risk_reward(self):
        """测试风险收益比验证"""
        entry = {'price': 15200, 'zone': (15190, 15210)}
        stop_loss = 15000
        take_profit = [15500, 15600, 15700]
        
        is_valid = self.generator._validate_risk_reward(entry, stop_loss, take_profit)
        self.assertTrue(is_valid)
        
    def test_assess_pattern_quality(self):
        """测试模式质量评估"""
        quality = self.generator._assess_pattern_quality(self.test_analysis)
        
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
    def test_assess_market_condition(self):
        """测试市场条件评估"""
        score = self.generator._assess_market_condition(self.test_analysis)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_assess_volume_profile(self):
        """测试成交量分析"""
        score = self.generator._assess_volume_profile(self.test_analysis)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效输入
        with self.assertRaises(ValueError):
            self.generator.generate_signal(None)
            
        # 测试缺失关键数据
        invalid_analysis = {'market_analysis': {}}
        with self.assertRaises(ValueError):
            self.generator.generate_signal(invalid_analysis)
            
    def tearDown(self):
        """测试后的清理"""
        self.generator = None

if __name__ == '__main__':
    unittest.main() 
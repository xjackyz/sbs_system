import unittest
import numpy as np
import cv2
from src.preprocessing.data_processor import DataProcessor, ProcessorConfig

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        self.config = ProcessorConfig(
            target_size=(224, 224),
            quality_threshold=0.8,
            brightness_range=(0.2, 0.8),
            contrast_range=(0.4, 0.8),
            noise_threshold=0.1
        )
        self.processor = DataProcessor(self.config)
        
        # 创建测试图像
        self.test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
        
    def test_image_preprocessing(self):
        """测试图像预处理"""
        result = self.processor.process_image(self.test_image)
        
        self.assertIsNotNone(result)
        self.assertIn('processed_image', result)
        self.assertEqual(result['processed_image'].shape[:2], self.config.target_size)
        
    def test_quality_assessment(self):
        """测试图像质量评估"""
        # 创建高质量图像
        high_quality = np.random.normal(128, 20, (300, 400, 3)).astype(np.uint8)
        result_high = self.processor._assess_quality(high_quality)
        
        # 创建低质量图像（噪声图像）
        low_quality = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result_low = self.processor._assess_quality(low_quality)
        
        self.assertGreater(result_high, result_low)
        
    def test_data_validation(self):
        """测试数据验证"""
        # 准备测试数据
        test_data = {
            'processed_image': self.test_image,
            'metadata': {
                'size': self.test_image.shape,
                'channels': 3,
                'quality_score': 0.9
            }
        }
        
        result = self.processor.validate_data(test_data)
        self.assertTrue(result['is_valid'])
        
    def test_image_enhancement(self):
        """测试图像增强"""
        # 创建暗图像
        dark_image = np.ones((300, 400, 3), dtype=np.uint8) * 50
        enhanced = self.processor._enhance_image(dark_image)
        
        # 检查亮度是否提高
        self.assertGreater(enhanced.mean(), dark_image.mean())
        
    def test_cache_mechanism(self):
        """测试缓存机制"""
        # 第一次处理
        result1 = self.processor.process_image(self.test_image)
        
        # 第二次处理相同图像
        result2 = self.processor.process_image(self.test_image)
        
        # 检查是否使用了缓存
        self.assertEqual(id(result1['processed_image']), id(result2['processed_image']))
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空图像
        with self.assertRaises(ValueError):
            self.processor.process_image(None)
            
        # 测试错误尺寸的图像
        invalid_image = np.ones((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.processor.process_image(invalid_image)
            
    def test_metadata_extraction(self):
        """测试元数据提取"""
        metadata = self.processor._extract_metadata(self.test_image)
        
        self.assertIn('size', metadata)
        self.assertIn('channels', metadata)
        self.assertIn('dtype', metadata)
        self.assertEqual(metadata['channels'], 3)
        
    def test_processing_steps_recording(self):
        """测试处理步骤记录"""
        self.processor.process_image(self.test_image)
        steps = self.processor._record_processing_steps()
        
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
        
    def tearDown(self):
        """测试后的清理"""
        self.processor = None

if __name__ == '__main__':
    unittest.main() 
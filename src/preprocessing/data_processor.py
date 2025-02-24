import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger('data_processor')

@dataclass
class ProcessorConfig:
    """数据处理器配置"""
    target_size: Tuple[int, int] = (224, 224)  # 目标图像尺寸
    quality_threshold: float = 0.8  # 图像质量阈值
    brightness_range: Tuple[float, float] = (0.2, 0.8)  # 亮度范围
    contrast_range: Tuple[float, float] = (0.4, 0.8)  # 对比度范围
    noise_threshold: float = 0.1  # 噪声阈值
    cache_size: int = 1000  # 缓存大小
    augmentation: Dict = None
    validation: Dict = None

class DataProcessor:
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        初始化数据处理器
        
        Args:
            config: 处理器配置
        """
        self.config = config or ProcessorConfig()
        self.cache = {}
        self.validation_history = []
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'cached_hits': 0
        }
        
    def process_image(self, image: np.ndarray) -> Dict:
        """
        处理输入的图表图像
        
        Args:
            image: 输入图像
            
        Returns:
            处理结果字典
        """
        try:
            # 1. 基础图像检查
            if not self._check_image_basic(image):
                raise ValueError("输入图像无效")
            
            # 2. 图像预处理
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                raise ValueError("图像预处理失败")
            
            # 3. 图像增强
            enhanced_image = self._enhance_image(processed_image)
            if enhanced_image is None:
                raise ValueError("图像增强失败")
            
            # 4. 提取元数据
            metadata = self._extract_metadata(enhanced_image)
            
            # 5. 质量评估
            quality_score = self._assess_quality(enhanced_image)
            
            # 6. 记录处理步骤
            steps_applied = self._record_processing_steps()
            
            # 7. 更新统计信息
            self.processing_stats['total_processed'] += 1
            if quality_score >= self.config.quality_threshold:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
            
            result = {
                'success': True,
                'image': enhanced_image,
                'metadata': metadata,
                'quality_score': quality_score,
                'processing_steps': steps_applied
            }
            
            # 更新缓存
            self._update_cache(result)
            
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            self.processing_stats['failed'] += 1
            return {'success': False, 'error': str(e)}
            
    def validate_data(self, processed_data: Dict) -> Dict:
        """
        验证处理后的数据
        
        Args:
            processed_data: 处理后的数据
            
        Returns:
            验证结果
        """
        try:
            validation_result = {
                'is_valid': True,
                'checks': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. 检查数据完整性
            completeness = self._check_data_completeness(processed_data)
            validation_result['checks']['completeness'] = completeness
            
            # 2. 验证图像质量
            if 'image' in processed_data:
                image_quality = self._validate_image_quality(processed_data['image'])
                validation_result['checks']['image_quality'] = image_quality
                
            # 3. 检查元数据
            if 'metadata' in processed_data:
                metadata_valid = self._validate_metadata(processed_data['metadata'])
                validation_result['checks']['metadata'] = metadata_valid
            
            # 4. 更新验证历史
            self._update_validation_history(validation_result)
            
            # 5. 设置最终验证结果
            validation_result['is_valid'] = all(validation_result['checks'].values())
            
            return validation_result
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {'is_valid': False, 'error': str(e)}
            
    def _check_image_basic(self, image: np.ndarray) -> bool:
        """检查图像基本属性"""
        if image is None or image.size == 0:
            return False
        if len(image.shape) != 3:
            return False
        if image.shape[2] != 3:  # 检查是否为RGB图像
            return False
        return True
        
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """图像预处理"""
        try:
            # 1. 调整大小
            resized = cv2.resize(image, self.config.target_size)
            
            # 2. 归一化
            normalized = resized / 255.0
            
            # 3. 去噪
            denoised = cv2.fastNlMeansDenoisingColored(
                (normalized * 255).astype(np.uint8),
                None,
                10,
                10,
                7,
                21
            )
            
            return denoised
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
        
    def _enhance_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """图像增强"""
        try:
            # 应用图像增强
            if self.config.augmentation:
                # 旋转
                if 'rotation_range' in self.config.augmentation:
                    angle = np.random.uniform(-self.config.augmentation['rotation_range'],
                                           self.config.augmentation['rotation_range'])
                    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                
                # 缩放
                if 'zoom_range' in self.config.augmentation:
                    zoom = np.random.uniform(1 - self.config.augmentation['zoom_range'],
                                          1 + self.config.augmentation['zoom_range'])
                    image = cv2.resize(image, None, fx=zoom, fy=zoom)
                    
                # 亮度调整
                if 'brightness_range' in self.config.augmentation:
                    alpha = np.random.uniform(self.config.augmentation['brightness_range'][0],
                                           self.config.augmentation['brightness_range'][1])
                    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                    
                # 对比度调整
                if 'contrast_range' in self.config.augmentation:
                    alpha = np.random.uniform(self.config.augmentation['contrast_range'][0],
                                           self.config.augmentation['contrast_range'][1])
                    image = cv2.convertScaleAbs(image, alpha=alpha, beta=128*(1-alpha))
            
            return image
            
        except Exception as e:
            logger.error(f"图像增强失败: {e}")
            return None
        
    def _extract_metadata(self, image: np.ndarray) -> Dict:
        """提取图像元数据"""
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'timestamp': datetime.now().isoformat()
        }
        
    def _assess_quality(self, image: np.ndarray) -> float:
        """评估图像质量"""
        try:
            scores = []
            
            # 亮度评分
            brightness = np.mean(image)
            brightness_score = self._calculate_range_score(
                brightness,
                self.config.brightness_range
            )
            scores.append(brightness_score)
            
            # 对比度评分
            contrast = np.std(image)
            contrast_score = self._calculate_range_score(
                contrast,
                self.config.contrast_range
            )
            scores.append(contrast_score)
            
            # 噪声评分
            noise_score = 1.0 - min(np.std(image) / self.config.noise_threshold, 1.0)
            scores.append(noise_score)
            
            # 清晰度评分
            laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
            clarity_score = min(laplacian / 500.0, 1.0)  # 500是经验值
            scores.append(clarity_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return 0.0
        
    def _calculate_range_score(self, value: float, target_range: Tuple[float, float]) -> float:
        """计算范围得分"""
        min_val, max_val = target_range
        if value < min_val:
            return max(0, value / min_val)
        elif value > max_val:
            return max(0, 1 - (value - max_val) / max_val)
        else:
            return 1.0
            
    def _record_processing_steps(self) -> List[str]:
        """记录处理步骤"""
        return [
            'basic_check',
            'preprocessing',
            'enhancement',
            'metadata_extraction',
            'quality_assessment'
        ]
        
    def _update_cache(self, result: Dict):
        """更新处理缓存"""
        cache_key = datetime.now().isoformat()
        if len(self.cache) >= self.config.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = result
        
    def _check_data_completeness(self, data: Dict) -> bool:
        """检查数据完整性"""
        required_fields = ['image', 'metadata', 'quality_score']
        return all(field in data for field in required_fields)
        
    def _validate_image_quality(self, image: np.ndarray) -> float:
        """验证图像质量"""
        return self._assess_quality(image)
        
    def _validate_metadata(self, metadata: Dict) -> bool:
        """验证元数据"""
        required_fields = ['shape', 'dtype', 'mean', 'std', 'min', 'max', 'timestamp']
        return all(field in metadata for field in required_fields)
        
    def _update_validation_history(self, result: Dict):
        """更新验证历史"""
        self.validation_history.append(result)
        if len(self.validation_history) > 1000:  # 保持历史记录在合理范围内
            self.validation_history = self.validation_history[-1000:]
            
    def get_processing_stats(self) -> Dict:
        """获取处理统计信息"""
        return {
            **self.processing_stats,
            'cache_size': len(self.cache),
            'validation_history_size': len(self.validation_history)
        } 
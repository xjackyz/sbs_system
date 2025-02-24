import cv2
import numpy as np
from PIL import Image
import os
import hashlib
from functools import lru_cache
import time

from config.config import MIN_IMAGE_QUALITY, IMAGE_RESIZE
from src.utils.logger import setup_logger

logger = setup_logger('image_processor')

class ImageProcessor:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 缓存超时时间（秒）
        self.max_cache_size = 100  # 最大缓存数量
    
    def _get_cache_key(self, image_path, operation):
        """生成缓存键"""
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{file_hash}_{operation}"
    
    def _clean_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [k for k, v in self.cache.items() 
                       if current_time - v['timestamp'] > self.cache_timeout]
        for k in expired_keys:
            del self.cache[k]
            
        # 如果缓存太大，删除最旧的条目
        if len(self.cache) > self.max_cache_size:
            sorted_cache = sorted(self.cache.items(), 
                                key=lambda x: x[1]['timestamp'])
            for k, _ in sorted_cache[:len(self.cache) - self.max_cache_size]:
                del self.cache[k]

    @staticmethod
    def check_image_quality(image_path):
        """
        检查图像质量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            bool: 图像质量是否合格
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # 计算图像清晰度（使用Laplacian算子）
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            quality_score = laplacian_var / 100000  # 归一化得分
            
            # 检查图像大小
            height, width = img.shape[:2]
            if height < 100 or width < 100:
                logger.warning(f"Image too small: {width}x{height}")
                return False
                
            # 检查图像是否过度压缩（通过分析高频分量）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            freq = np.fft.fft2(gray)
            freq_shift = np.fft.fftshift(freq)
            magnitude = np.abs(freq_shift)
            high_freq_ratio = np.sum(magnitude > np.mean(magnitude)) / magnitude.size
            
            # 检查图像对比度
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            contrast = np.std(hist_norm)
            
            quality_ok = (
                quality_score >= MIN_IMAGE_QUALITY and 
                high_freq_ratio >= 0.1 and
                contrast >= 0.1
            )
            
            if not quality_ok:
                logger.warning(
                    f"Low image quality: score={quality_score:.2f}, "
                    f"high_freq_ratio={high_freq_ratio:.2f}, "
                    f"contrast={contrast:.2f}"
                )
            
            return quality_ok
            
        except Exception as e:
            logger.error(f"Error checking image quality: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 处理后的图像路径，如果处理失败则返回None
        """
        try:
            # 检查缓存
            cache_key = self._get_cache_key(image_path, 'preprocess')
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] <= self.cache_timeout:
                    logger.info("Using cached preprocessed image")
                    return cache_entry['result']
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image for preprocessing: {image_path}")
                return None
            
            # 调整大小
            img = cv2.resize(img, IMAGE_RESIZE)
            
            # 增强对比度
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 降噪
            denoised = cv2.fastNlMeansDenoisingColored(enhanced)
            
            # 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 保存处理后的图像
            output_path = image_path.replace('.', '_processed.')
            cv2.imwrite(output_path, sharpened)
            
            # 更新缓存
            self.cache[cache_key] = {
                'result': output_path,
                'timestamp': time.time()
            }
            self._clean_expired_cache()
            
            logger.info(f"Image preprocessed and saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def crop_chart_area(self, image_path):
        """
        裁剪图表区域，去除无关内容
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 裁剪后的图像路径，如果处理失败则返回None
        """
        try:
            # 检查缓存
            cache_key = self._get_cache_key(image_path, 'crop')
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] <= self.cache_timeout:
                    logger.info("Using cached cropped image")
                    return cache_entry['result']
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image for cropping: {image_path}")
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found in the image")
                return image_path
                
            # 找到最大的轮廓（假设是图表区域）
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 添加边距
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # 裁剪图像
            cropped = img[y:y+h, x:x+w]
            
            # 保存裁剪后的图像
            output_path = image_path.replace('.', '_cropped.')
            cv2.imwrite(output_path, cropped)
            
            # 更新缓存
            self.cache[cache_key] = {
                'result': output_path,
                'timestamp': time.time()
            }
            self._clean_expired_cache()
            
            logger.info(f"Image cropped and saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            return None 
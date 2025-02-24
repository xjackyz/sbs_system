import os
import sys
import psutil
import torch
import gc
from typing import Dict, List, Optional
from pathlib import Path

from src.utils.logger import setup_logger
from config.config import (
    LLAVA_MODEL_PATH,
    CLIP_MODEL_PATH,
    LOG_DIR,
    SCREENSHOT_DIR,
    IMAGE_RESIZE,
    CLIP_IMAGE_SIZE
)

logger = setup_logger('system_validator')

class SystemValidator:
    """系统验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.required_paths = [
            LLAVA_MODEL_PATH,
            LOG_DIR,
            SCREENSHOT_DIR
        ]
        
        self.resource_thresholds = {
            'gpu_memory': 0.9,  # 90%显存使用率警告
            'cpu_memory': 0.8,  # 80%内存使用率警告
            'disk_space': 0.9   # 90%磁盘使用率警告
        }
        
        self.min_gpu_memory = 8  # GB
        
    def validate_system(self) -> bool:
        """
        验证系统环境
        
        Returns:
            bool: 系统是否满足运行要求
        """
        try:
            checks = [
                self._check_paths(),
                self._check_gpu(),
                self._check_resources(),
                self._check_config_compatibility()
            ]
            
            return all(checks)
            
        except Exception as e:
            logger.error(f"系统验证失败: {e}")
            return False
    
    def _check_paths(self) -> bool:
        """检查必要路径"""
        try:
            for path in self.required_paths:
                if not path:
                    logger.error(f"路径未设置: {path}")
                    return False
                    
                if not os.path.exists(path):
                    logger.error(f"路径不存在: {path}")
                    return False
            
            logger.info("路径检查通过")
            return True
            
        except Exception as e:
            logger.error(f"路径检查失败: {e}")
            return False
    
    def _check_gpu(self) -> bool:
        """检查GPU环境"""
        try:
            if not torch.cuda.is_available():
                logger.error("未检测到可用的GPU")
                return False
            
            # 检查GPU数量
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.error("未找到GPU设备")
                return False
            
            # 检查每个GPU的显存
            for i in range(gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                if gpu_memory < self.min_gpu_memory:
                    logger.error(f"GPU {i} 显存不足: {gpu_memory:.1f}GB < {self.min_gpu_memory}GB")
                    return False
                logger.info(f"GPU {i}: {gpu_memory:.1f}GB 显存")
            
            logger.info("GPU环境检查通过")
            return True
            
        except Exception as e:
            logger.error(f"GPU检查失败: {e}")
            return False
    
    def _check_resources(self) -> bool:
        """检查系统资源"""
        try:
            # 检查CPU内存
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            if memory_usage > self.resource_thresholds['cpu_memory']:
                logger.warning(f"内存使用率过高: {memory_usage:.1%}")
            
            # 检查磁盘空间
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100
            if disk_usage > self.resource_thresholds['disk_space']:
                logger.warning(f"磁盘使用率过高: {disk_usage:.1%}")
            
            # 检查GPU显存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
                    if memory_allocated > self.resource_thresholds['gpu_memory']:
                        logger.warning(f"GPU {i} 显存使用率过高: {memory_allocated:.1%}")
            
            logger.info("资源检查完成")
            return True
            
        except Exception as e:
            logger.error(f"资源检查失败: {e}")
            return False
    
    def _check_config_compatibility(self) -> bool:
        """检查配置兼容性"""
        try:
            # 检查图像尺寸配置
            if IMAGE_RESIZE[0] != CLIP_IMAGE_SIZE or IMAGE_RESIZE[1] != CLIP_IMAGE_SIZE:
                logger.warning(f"图像尺寸配置与CLIP不匹配: {IMAGE_RESIZE} != ({CLIP_IMAGE_SIZE}, {CLIP_IMAGE_SIZE})")
            
            # 检查环境变量
            required_env_vars = [
                'LLAVA_MODEL_PATH',
                'DISCORD_WEBHOOK_MONITOR',
                'DISCORD_WEBHOOK_SIGNAL',
                'DISCORD_WEBHOOK_DEBUG'
            ]
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                logger.error(f"缺少环境变量: {', '.join(missing_vars)}")
                return False
            
            logger.info("配置兼容性检查通过")
            return True
            
        except Exception as e:
            logger.error(f"配置兼容性检查失败: {e}")
            return False
    
    def monitor_resources(self) -> Dict:
        """
        监控系统资源使用情况
        
        Returns:
            Dict: 资源使用情况统计
        """
        try:
            stats = {
                'cpu': {
                    'usage': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total': psutil.virtual_memory().total / (1024**3),  # GB
                    'used': psutil.virtual_memory().used / (1024**3),    # GB
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total': psutil.disk_usage('/').total / (1024**3),   # GB
                    'used': psutil.disk_usage('/').used / (1024**3),     # GB
                    'percent': psutil.disk_usage('/').percent
                },
                'gpu': []
            }
            
            # GPU统计
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_stats = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                        'allocated_memory': torch.cuda.memory_allocated(i) / (1024**3),                # GB
                        'cached_memory': torch.cuda.memory_reserved(i) / (1024**3),                    # GB
                        'utilization': torch.cuda.utilization(i)
                    }
                    stats['gpu'].append(gpu_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"资源监控失败: {e}")
            return {}
    
    def cleanup_resources(self):
        """清理系统资源"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 清理临时文件
            temp_dirs = ['screenshots/temp', 'logs/temp']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except Exception as e:
                            logger.warning(f"清理临时文件失败: {e}")
            
            logger.info("系统资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def get_system_info(self) -> Dict:
        """
        获取系统信息
        
        Returns:
            Dict: 系统信息摘要
        """
        try:
            info = {
                'platform': {
                    'system': sys.platform,
                    'python_version': sys.version,
                    'torch_version': torch.__version__
                },
                'hardware': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                    'disk_total': f"{psutil.disk_usage('/').total / (1024**3):.1f}GB"
                },
                'gpu': []
            }
            
            # GPU信息
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        'name': torch.cuda.get_device_name(i),
                        'memory': f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f}GB"
                    }
                    info['gpu'].append(gpu_info)
            
            return info
            
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {} 
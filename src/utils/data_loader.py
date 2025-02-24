import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json

logger = logging.getLogger(__name__)

class OptimizedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        cache_size: int = 1000,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        优化的数据集类
        
        Args:
            data_dir: 数据目录
            transform: 数据转换函数
            cache_size: 缓存大小
            num_workers: 数据加载线程数
            pin_memory: 是否使用固定内存
        """
        self.data_dir = data_dir
        self.transform = transform
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 初始化缓存
        self.cache = {}
        self.cache_keys = []
        
        # 加载数据索引
        self.data_index = self._load_data_index()
        
    def _load_data_index(self) -> List[Dict]:
        """加载数据索引"""
        index_file = os.path.join(self.data_dir, 'index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                return json.load(f)
        return []
        
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图像"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            return None
            
    def _load_label(self, label_path: str) -> Optional[Dict]:
        """加载标签"""
        try:
            with open(label_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载标签失败 {label_path}: {e}")
            return None
            
    def _update_cache(self, key: str, value: Tuple):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最早的缓存
            old_key = self.cache_keys.pop(0)
            del self.cache[old_key]
            
        self.cache[key] = value
        self.cache_keys.append(key)
        
    def __len__(self) -> int:
        return len(self.data_index)
        
    def __getitem__(self, idx: int) -> Tuple:
        """获取数据项"""
        data_info = self.data_index[idx]
        cache_key = str(idx)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # 加载数据
        image_path = os.path.join(self.data_dir, data_info['image'])
        label_path = os.path.join(self.data_dir, data_info['label'])
        
        image = self._load_image(image_path)
        label = self._load_label(label_path)
        
        if image is None or label is None:
            # 返回空数据
            return None, None
            
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        # 更新缓存
        result = (image, label)
        self._update_cache(cache_key, result)
        
        return result

class OptimizedDataLoader:
    def __init__(
        self,
        dataset: OptimizedDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        """
        优化的数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            pin_memory: 是否使用固定内存
            prefetch_factor: 预取因子
            persistent_workers: 是否使用持久化工作线程
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # 创建数据加载器
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )
        
    def __iter__(self):
        return iter(self.loader)
        
    def __len__(self):
        return len(self.loader)
        
class DynamicBatchSizeFinder:
    def __init__(
        self,
        dataset: OptimizedDataset,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        memory_threshold: float = 0.8
    ):
        """
        动态批处理大小查找器
        
        Args:
            dataset: 数据集
            min_batch_size: 最小批处理大小
            max_batch_size: 最大批处理大小
            memory_threshold: 内存使用率阈值
        """
        self.dataset = dataset
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        
    def _test_batch_size(self, batch_size: int) -> bool:
        """测试批处理大小是否可用"""
        try:
            # 创建临时数据加载器
            loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                num_workers=1,
                pin_memory=True
            )
            
            # 尝试加载一个批次
            for batch in loader:
                break
                
            # 检查内存使用率
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if memory_used > self.memory_threshold:
                    return False
                    
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
            
    def find_optimal_batch_size(self) -> int:
        """
        查找最优批处理大小
        
        Returns:
            最优批处理大小
        """
        logger.info("开始查找最优批处理大小...")
        
        left = self.min_batch_size
        right = self.max_batch_size
        optimal_batch_size = self.min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            if self._test_batch_size(mid):
                optimal_batch_size = mid
                left = mid + 1
            else:
                right = mid - 1
                
        logger.info(f"找到最优批处理大小: {optimal_batch_size}")
        return optimal_batch_size

def create_optimized_data_loader(
    data_dir: str,
    batch_size: Optional[int] = None,
    transform=None,
    cache_size: int = 1000,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    find_batch_size: bool = True
) -> OptimizedDataLoader:
    """
    创建优化的数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批处理大小
        transform: 数据转换函数
        cache_size: 缓存大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存
        prefetch_factor: 预取因子
        persistent_workers: 是否使用持久化工作线程
        find_batch_size: 是否自动查找最优批处理大小
        
    Returns:
        优化的数据加载器
    """
    # 创建数据集
    dataset = OptimizedDataset(
        data_dir=data_dir,
        transform=transform,
        cache_size=cache_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 查找最优批处理大小
    if find_batch_size and batch_size is None:
        finder = DynamicBatchSizeFinder(dataset)
        batch_size = finder.find_optimal_batch_size()
    elif batch_size is None:
        batch_size = 32  # 默认批处理大小
        
    # 创建数据加载器
    return OptimizedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    ) 
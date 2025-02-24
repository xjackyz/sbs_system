"""
图表标注工具
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional
from pathlib import Path
from config.config import SBS_ANNOTATION_CONFIG
from ..utils.logger import setup_logger

logger = setup_logger('chart_annotator')

class ChartAnnotator:
    def __init__(self, config):
        """初始化图表标注器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.annotation_config = SBS_ANNOTATION_CONFIG
    
    async def annotate_signal(self, 
                            image_path: str, 
                            analysis_result: dict) -> str:
        """仅在有效交易信号时标注图表
        
        Args:
            image_path: 原始图表路径
            analysis_result: LLaVA分析结果
            
        Returns:
            str: 标注后的图片路径
        """
        try:
            # 检查是否是有效信号
            if not self._is_valid_signal(analysis_result):
                logger.info("非有效交易信号，跳过标注")
                return image_path
            
            # 读取图片
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 获取价格范围
            price_range = self._get_price_range(analysis_result)
            if price_range is None:
                raise ValueError("无法确定价格范围")
            
            # 标注关键点
            key_points = analysis_result.get('key_points', {})
            for point_type, point_data in key_points.items():
                if point_type in self.annotation_config['points']:
                    config = self.annotation_config['points'][point_type]
                    if point_data is not None:  # 确保价格值存在
                        # 转换价格水平为图像坐标
                        y_coord = self._price_to_y_coordinate(
                            float(point_data), 
                            image.shape[0],
                            price_range
                        )
                        # 添加标记点和标签
                        self._add_point_marker(
                            image, 
                            y_coord,
                            config['name'],
                            config['color'],
                            config['marker_size']
                        )
            
            # 添加趋势线
            self._add_trend_lines(image, analysis_result)
            
            # 添加置信度和风险等级
            self._add_analysis_info(image, analysis_result)
            
            # 保存标注后的图片
            output_path = self._get_output_path(image_path)
            cv2.imwrite(output_path, image)
            
            logger.info(f"图表标注完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"图表标注失败: {str(e)}")
            return image_path
            
    def _is_valid_signal(self, analysis_result: dict) -> bool:
        """检查是否是有效信号
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            bool: 是否有效
        """
        sequence_eval = analysis_result.get('sequence_evaluation', {})
        return (sequence_eval.get('validity') == '是' and
                sequence_eval.get('confidence', 0) > 70)
                
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            Optional[np.ndarray]: 图片数据
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            return image
        except Exception as e:
            logger.error(f"图片加载失败: {e}")
            return None
            
    def _get_price_range(self, analysis_result: dict) -> Optional[Tuple[float, float]]:
        """获取价格范围
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            Optional[Tuple[float, float]]: (最小价格, 最大价格)
        """
        try:
            key_points = analysis_result.get('key_points', {})
            prices = [float(p) for p in key_points.values() if p is not None]
            if not prices:
                return None
            
            # 添加一定的边界空间
            margin = (max(prices) - min(prices)) * 0.1
            return (min(prices) - margin, max(prices) + margin)
            
        except Exception as e:
            logger.error(f"价格范围计算失败: {e}")
            return None
    
    def _price_to_y_coordinate(self, 
                             price: float, 
                             image_height: int,
                             price_range: Tuple[float, float]) -> int:
        """将价格转换为图像Y坐标
        
        Args:
            price: 价格值
            image_height: 图像高度
            price_range: 价格范围
            
        Returns:
            int: Y坐标
        """
        min_price, max_price = price_range
        y_ratio = (max_price - price) / (max_price - min_price)
        return int(y_ratio * image_height)
    
    def _add_point_marker(self, 
                         image: np.ndarray,
                         y_coord: int,
                         label: str,
                         color: tuple,
                         size: int):
        """添加标记点和标签
        
        Args:
            image: 图像数据
            y_coord: Y坐标
            label: 标签文本
            color: 颜色
            size: 标记点大小
        """
        height, width = image.shape[:2]
        x_coord = width - 100  # 在右侧标注
        
        # 画点
        cv2.circle(image, (x_coord, y_coord), size, color, -1)
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            label,
            (x_coord + 15, y_coord + 5),
            font,
            self.annotation_config['font']['size'] / 30,
            self.annotation_config['font']['color'],
            self.annotation_config['font']['thickness']
        )
        
    def _add_trend_lines(self, 
                        image: np.ndarray, 
                        analysis_result: dict):
        """添加趋势线
        
        Args:
            image: 图像数据
            analysis_result: 分析结果
        """
        # TODO: 实现趋势线绘制
        pass
        
    def _add_analysis_info(self, 
                          image: np.ndarray, 
                          analysis_result: dict):
        """添加分析信息
        
        Args:
            image: 图像数据
            analysis_result: 分析结果
        """
        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 添加置信度
        confidence = analysis_result.get('sequence_evaluation', {}).get('confidence', 0)
        cv2.putText(
            image,
            f"置信度: {confidence}%",
            (10, 30),
            font,
            0.7,
            self.annotation_config['font']['color'],
            self.annotation_config['font']['thickness']
        )
        
        # 添加风险等级
        risk_level = analysis_result.get('risk_assessment', {}).get('risk_level', '未知')
        cv2.putText(
            image,
            f"风险等级: {risk_level}",
            (10, 60),
            font,
            0.7,
            self.annotation_config['font']['color'],
            self.annotation_config['font']['thickness']
        )
        
    def _get_output_path(self, image_path: str) -> str:
        """获取输出路径
        
        Args:
            image_path: 原始图片路径
            
        Returns:
            str: 输出路径
        """
        path = Path(image_path)
        return str(path.parent / f"{path.stem}_annotated{path.suffix}") 
import os
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List
import pandas as pd
import sys
import torch
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.llava_analyzer import LLaVAAnalyzer
from src.utils.trading_alerts import TradingAlertManager
from config.config import LLAVA_PROMPT_TEMPLATE
from src.config import Config

class BatchAnalyzer:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化批量分析器
        
        Args:
            model_path: 模型路径
            device: 运行设备
        """
        self.logger = logging.getLogger(__name__)
        
        # 创建配置对象
        config = Config(
            model_path=model_path,
            device=device
        )
        
        # 初始化LLaVA分析器
        self.model = LLaVAAnalyzer(config)
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            self.logger.info(f"开始处理图片: {image_path}")
            result = self.model.analyze_image(image_path, LLAVA_PROMPT_TEMPLATE)
            
            return {
                'image_path': image_path,
                'analysis': result,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            self.logger.error(f"处理图片 {os.path.basename(image_path)} 时出错: {str(e)}")
            self.logger.error("错误详情:", exc_info=True)
            
            return {
                'image_path': image_path,
                'error': str(e),
                'status': 'error'
            }
            
    def analyze_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        分析目录中的所有图片
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        results = []
        image_files = []
        
        # 获取所有图片文件
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(input_dir, filename))
                
        # 使用tqdm显示进度
        for i, img_path in enumerate(tqdm(image_files, desc="处理图片")):
            self.logger.info(f"正在处理图片 [{i+1}/{len(image_files)}]: {os.path.basename(img_path)}")
            result = self.process_image(img_path)
            if result['status'] == 'success':
                results.append(result)
                
        return results
        
    def generate_report(self, results: List[Dict[str, Any]], output_format: str = 'json') -> str:
        """
        生成分析报告
        
        Args:
            results: 分析结果列表
            output_format: 输出格式
            
        Returns:
            str: 报告文件路径
        """
        if not results:
            self.logger.warning("没有有效的分析结果")
            return ""
            
        # 统计信息
        stats = {
            'total_images': len(results),
            'successful_analyses': len([r for r in results if r['status'] == 'success']),
            'failed_analyses': len([r for r in results if r['status'] == 'error']),
            'average_confidence': sum(
                float(r.get('confidence', 0)) 
                for r in results 
                if r['status'] == 'success' and 'confidence' in r
            ) / len([r for r in results if r['status'] == 'success' and 'confidence' in r]) if results else 0
        }
        
        # 准备输出数据
        output_data = {
            'statistics': stats,
            'results': results
        }
        
        # 生成输出文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'output/analysis_report_{timestamp}.{output_format}'
        
        # 确保输出目录存在
        os.makedirs('output', exist_ok=True)
        
        # 写入文件
        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
            
        self.logger.info(f"报告已生成: {output_file}")
        return output_file
        
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印分析摘要"""
        print("\n=== SBS序列分析摘要 ===")
        print(f"总图片数: {len(results)}")
        
        # 序列类型分布
        sequence_types = {}
        for result in results:
            signal_type = result['trading_signal']['方向']
            sequence_types[signal_type] = sequence_types.get(signal_type, 0) + 1
            
        print("\n序列类型分布:")
        for signal_type, count in sequence_types.items():
            print(f"{signal_type}: {count} ({count/len(results)*100:.1f}%)")
            
        # 置信度统计
        confidences = [float(r['sequence_evaluation']['可信度'].rstrip('%')) for r in results]
        avg_conf = sum(confidences) / len(confidences)
        high_conf = len([c for c in confidences if c >= 80])
        
        print(f"\n平均置信度: {avg_conf:.2f}%")
        print(f"高置信度预测 (>=80%): {high_conf} ({high_conf/len(results)*100:.1f}%)")
        
        # 风险等级分布
        risk_levels = {}
        for result in results:
            risk = result['risk_assessment']['风险等级']
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
            
        print("\n风险等级分布:")
        for risk, count in risk_levels.items():
            print(f"{risk}: {count} ({count/len(results)*100:.1f}%)")
            
        # 交易提醒
        alerts = [r for r in results if r.get('alert')]
        if alerts:
            print("\n生成的交易提醒:")
            for alert in alerts:
                print(f"- {alert['file_name']}: {alert['trading_signal']['方向']} "
                      f"(置信度: {alert['sequence_evaluation']['可信度']})") 
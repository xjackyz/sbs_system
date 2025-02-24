import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from typing import Dict, List
import json
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch.cuda.amp as amp
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.model.llava_analyzer import LLaVAAnalyzer
from src.self_supervised.data_generator.sequence_generator import SequenceGenerator
from src.self_supervised.validator.performance_validator import PerformanceValidator
from scripts.init_workspace import init_workspace
from config.config import (
    SEQUENCE_CONFIRMATION,
    TRAINING_OPTIMIZATION,
    GPU_OPTIMIZATION,
    DISTRIBUTED_TRAINING,
    OPTIMIZER_CONFIG,
    MODEL_CONFIG
)

logger = setup_logger('full_year_training')

def setup_distributed():
    """设置分布式训练"""
    if DISTRIBUTED_TRAINING['enabled']:
        dist.init_process_group(
            backend=DISTRIBUTED_TRAINING['backend'],
            world_size=DISTRIBUTED_TRAINING['world_size'],
            rank=DISTRIBUTED_TRAINING['rank']
        )

def cleanup_distributed():
    """清理分布式训练"""
    if DISTRIBUTED_TRAINING['enabled']:
        dist.destroy_process_group()

def setup_optimizer(model):
    """设置优化器和学习率调度器"""
    optimizer = AdamW(
        model.parameters(),
        lr=OPTIMIZER_CONFIG['lr'],
        weight_decay=OPTIMIZER_CONFIG['weight_decay']
    )
    
    scheduler = OneCycleLR(
        optimizer,
        **OPTIMIZER_CONFIG['scheduler']
    )
    
    return optimizer, scheduler

def setup_dataloader(dataset):
    """设置数据加载器"""
    return DataLoader(
        dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        num_workers=TRAINING_OPTIMIZATION['num_workers'],
        pin_memory=TRAINING_OPTIMIZATION['pin_memory'],
        prefetch_factor=TRAINING_OPTIMIZATION['prefetch_factor']
    )

async def analyze_sequence(llava_analyzer: LLaVAAnalyzer, sequence: Dict) -> Dict:
    """分析单个序列"""
    try:
        # 使用LLaVA分析序列
        analysis = await llava_analyzer.analyze_chart(sequence['chart_path'])
        
        # 验证分析结果
        validation = {
            'breakout_correct': False,
            'sbs_sequence_valid': False,
            'confirmation_signals': {
                'sce': False,
                'double_pattern': False,
                'liquidation': False
            },
            'entry_exit_valid': False,
            'trend_correct': False
        }
        
        # 验证突破识别
        if analysis.get('breakout', {}).get('detected'):
            validation['breakout_correct'] = True
            logger.info(f"序列 {sequence['id']} 突破识别正确")
        
        # 验证SBS序列
        if analysis.get('sequence', {}).get('detected'):
            validation['sbs_sequence_valid'] = True
            logger.info(f"序列 {sequence['id']} SBS序列识别正确")
        
        # 验证确认信号
        signals = analysis.get('confirmation_signals', {})
        for signal_type in ['sce', 'double_pattern', 'liquidation']:
            if signals.get(signal_type, {}).get('detected'):
                validation['confirmation_signals'][signal_type] = True
                logger.info(f"序列 {sequence['id']} {signal_type}信号识别正确")
        
        # 验证入场出场点
        trade_signal = analysis.get('trade_signal', {})
        if trade_signal.get('entry_price') and trade_signal.get('stop_loss') and trade_signal.get('take_profit'):
            validation['entry_exit_valid'] = True
            logger.info(f"序列 {sequence['id']} 入场出场点识别正确")
            
            # 计算盈利点数
            entry_price = float(trade_signal['entry_price'])
            take_profit = float(trade_signal['take_profit'])
            
            # 根据序列类型计算盈利点数
            if analysis.get('sequence', {}).get('type') == 'upward':
                profit_points = take_profit - entry_price
            else:  # downward
                profit_points = entry_price - take_profit
                
            trade_signal['profit_points'] = profit_points
            logger.info(f"序列 {sequence['id']} 潜在盈利点数: {profit_points:.1f}")
        
        # 验证趋势判断
        if analysis.get('trend_analysis', {}).get('trend'):
            validation['trend_correct'] = True
            logger.info(f"序列 {sequence['id']} 趋势判断正确")
        
        return {
            'sequence': sequence,
            'analysis': analysis,
            'validation': validation
        }
        
    except Exception as e:
        logger.error(f"序列 {sequence['id']} 分析失败: {e}")
        return None

def monitor_gpu_memory():
    """监控GPU内存使用"""
    if GPU_OPTIMIZATION['memory_monitoring']:
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_percent = memory_allocated / memory_total * 100
        
        if memory_percent > GPU_OPTIMIZATION['cache_cleanup_threshold'] * 100:
            logger.warning(f"GPU内存使用率过高: {memory_percent:.1f}%")
            torch.cuda.empty_cache()
            gc.collect()
            
        return memory_percent
    return 0

async def train_full_year():
    """训练2024年全年数据"""
    try:
        # 设置CUDA内存分配器
        if GPU_OPTIMIZATION['memory_growth']:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{GPU_OPTIMIZATION['max_split_size_mb']}"
        torch.cuda.empty_cache()
        
        logger.info("正在设置训练环境...")
        
        # 设置多进程环境
        num_cores = mp.cpu_count()
        torch.set_num_threads(num_cores)
        logger.info(f"设置CPU线程数: {num_cores}")
        
        # 设置分布式训练
        if DISTRIBUTED_TRAINING['enabled']:
            setup_distributed()
            logger.info("分布式训练已启用")
        
        # 启用梯度检查点和混合精度训练
        torch.backends.cudnn.benchmark = True
        scaler = amp.GradScaler('cuda')  # 修复废弃警告
        logger.info("混合精度训练已启用")
        
        # 初始化工作空间
        logger.info("初始化工作空间...")
        init_workspace()
        
        # 初始化LLaVA分析器
        logger.info("初始化LLaVA分析器...")
        llava_analyzer = LLaVAAnalyzer()
        
        # 如果启用torch.compile
        if TRAINING_OPTIMIZATION['torch_compile']:
            logger.info("正在编译模型...")
            llava_analyzer.model = torch.compile(llava_analyzer.model)
            logger.info("模型编译完成")
        
        # 设置优化器和调度器
        logger.info("设置优化器和学习率调度器...")
        optimizer, scheduler = setup_optimizer(llava_analyzer.model)
        
        # 设置时间范围
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        logger.info(f"训练时间范围: {start_date} 到 {end_date}")
        
        # 按月处理数据
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 性能统计
        monthly_stats = {}
        
        while current_date <= end_datetime:
            month_start = current_date.strftime("%Y-%m-%d")
            month_end = (current_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            month_end = min(month_end, end_datetime).strftime("%Y-%m-%d")
            
            logger.info(f"\n开始处理 {month_start} 到 {month_end} 的数据")
            
            try:
                # 检查GPU内存
                memory_percent = monitor_gpu_memory()
                logger.info(f"当前GPU内存使用率: {memory_percent:.1f}%")
                
                # 生成序列数据
                logger.info("正在生成序列数据...")
                sequence_generator = SequenceGenerator()
                sequences = sequence_generator.generate_training_data(month_start, month_end)
                logger.info(f"生成了 {len(sequences)} 个训练序列")
                
                # 创建数据加载器
                dataloader = setup_dataloader(sequences)
                logger.info(f"创建数据加载器，批次大小: {MODEL_CONFIG['batch_size']}")
                
                # 梯度累积步数
                accum_steps = TRAINING_OPTIMIZATION['gradient_accumulation']
                
                # 分批处理序列
                total_batches = len(dataloader)
                logger.info(f"开始训练，总批次数: {total_batches}")
                
                for i, batch in enumerate(dataloader):
                    # 检查GPU内存
                    if monitor_gpu_memory() > GPU_OPTIMIZATION['cache_cleanup_threshold'] * 100:
                        logger.warning("等待GPU内存释放...")
                        time.sleep(5)
                        continue
                    
                    batch_start_time = time.time()
                    
                    # 使用混合精度训练
                    with amp.autocast():
                        # 并行分析序列
                        with ThreadPoolExecutor(max_workers=num_cores) as pool:
                            analysis_futures = [
                                pool.submit(
                                    analyze_sequence, 
                                    llava_analyzer, 
                                    sequence
                                ) for sequence in batch
                            ]
                            analysis_results = [
                                future.result() for future in analysis_futures 
                                if future.result() is not None
                            ]
                        
                        # 计算损失
                        loss = llava_analyzer.compute_loss(analysis_results)
                        loss = loss / accum_steps
                        
                        # 反向传播
                        scaler.scale(loss).backward()
                        
                        # 梯度累积
                        if (i + 1) % accum_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()
                            
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.info(f"批次 [{i+1}/{total_batches}] - "
                                      f"损失: {loss.item():.4f}, "
                                      f"学习率: {current_lr:.6f}, "
                                      f"用时: {time.time() - batch_start_time:.2f}秒")
                    
                    # 强制GPU同步和内存清理
                    torch.cuda.synchronize()
                    monitor_gpu_memory()
                    
                    # 验证性能
                    if (i + 1) % 10 == 0:  # 每10个批次验证一次
                        logger.info("正在验证性能...")
                        validator = PerformanceValidator()
                        validation_results = validator.validate_performance(analysis_results)
                        
                        # 记录性能指标
                        monthly_stats[month_start] = validation_results
                        logger.info(f"验证结果: {validation_results}")
                    
                    # 清理内存
                    del analysis_futures
                    del analysis_results
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"完成 {month_start} 月份的训练")
                
            except Exception as e:
                logger.error(f"处理月份数据出错: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            
            # 移动到下一个月
            current_date = (current_date + timedelta(days=32)).replace(day=1)
            
            # 每月结束后强制清理内存
            gc.collect()
            torch.cuda.empty_cache()
        
        # 清理分布式训练
        if DISTRIBUTED_TRAINING['enabled']:
            cleanup_distributed()
        
        logger.info("训练完成!")
        logger.info(f"月度性能统计: {monthly_stats}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        sys.exit(1)
    finally:
        # 清理所有资源
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join("logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
        
        # 设置多进程启动方法
        if sys.platform != 'darwin':  # 非Mac系统使用spawn
            mp.set_start_method('spawn', force=True)
        
        # 运行训练
        asyncio.run(train_full_year())
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise
    finally:
        # 清理资源
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
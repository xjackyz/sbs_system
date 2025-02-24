"""
系统配置文件
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import logging

# 加载环境变量
load_dotenv()

# 路径配置
PATHS = {
    'base_dir': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_dir': 'data',
    'log_dir': 'logs',
    'model_dir': 'models',
    'cache_dir': 'cache',
    'temp_dir': 'temp',
    'backup_dir': 'backup',
    'results_dir': 'results',
    'reports_dir': 'reports',
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'validation_data': 'data/validation',
    'checkpoints': 'models/checkpoints',
    'screenshots': 'screenshots'
}

# 确保所有路径存在
for path in PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

# 截图和模型路径
SCREENSHOT_DIR = PATHS['screenshots']
CLIP_MODEL_PATH = os.getenv('CLIP_MODEL_PATH', 'models/clip-sbs')

# LLaVA模型配置
LLAVA_MODEL_PATH = os.getenv('LLAVA_MODEL_PATH', 'models/llava-sbs')

# SBS分析Prompt配置
SBS_PROMPT = {
    'role_definition': """你是一个专业的金融图表分析专家，专注于识别和分析SBS(Sequence Based Signal)交易序列。
    你需要分析图表中的：
    1. SBS序列的完整性和有效性
    2. 关键点位的位置和重要性
    3. 交易信号的生成和确认
    4. 市场结构和趋势状态""",
    
    'task_focus': """请分析这张图表并关注：
    - 突破位置和有效性
    - 回调的深度和形态
    - 双顶/双底的形成
    - 流动性获取区域
    - SCE信号确认
    - SMA20和SMA200趋势辅助""",
    
    'analysis_requirements': {
        'sequence_validation': """对于SBS序列，请确认：
    1. 突破的清晰度和有效性
    2. 回调的规范性（0.382-0.618）
    3. 确认阶段的完整性
    4. 整体序列的时间结构""",
        
        'key_points': """请标识以下关键点位：
    1. 突破点
    2. Point 1 (回调高点)
    3. Point 2 (回调低点)
    4. Point 3 (确认高点)
    5. Point 4 (确认低点)""",
        
        'signal_generation': """请提供：
    1. 信号类型（做多/做空）
    2. 入场区域建议
    3. 止损位置
    4. 目标位置"""
    },
    
    'output_format': """请按以下格式输出分析结果：

序列评估：
- 有效性：[是/否]
- 完整度：[0-100%]
- 可信度：[0-100%]

关键点位：
- 突破点：[价格水平]
- Point 1：[价格水平]
- Point 2：[价格水平]
- Point 3：[价格水平]
- Point 4：[价格水平]

交易信号：
- 方向：[多/空]
- 入场区域：[价格范围]
- 止损位：[价格水平]
- 目标位：[价格水平]

趋势分析：
- SMA20趋势：[上升/下降/盘整]
- SMA200趋势：[上升/下降/盘整]
- 整体趋势评估：[描述]

风险评估：
- 风险等级：[低/中/高]
- 主要风险点：[描述]"""
}

# 更新LLAVA_PROMPT_TEMPLATE
LLAVA_PROMPT_TEMPLATE = f"""
{SBS_PROMPT['role_definition']}

{SBS_PROMPT['task_focus']}

分析要求：
{SBS_PROMPT['analysis_requirements']['sequence_validation']}
{SBS_PROMPT['analysis_requirements']['key_points']}
{SBS_PROMPT['analysis_requirements']['signal_generation']}

请按照以下格式输出：
{SBS_PROMPT['output_format']}

注意事项：
1. 只在确认看到清晰的SBS序列时才生成信号
2. 对于不完整或不确定的形态，请明确指出原因
3. 如果发现任何潜在风险，请在风险评估中详细说明
4. 所有价格水平必须精确到小数点后4位
5. 确保所有关键点位的时间顺序正确
"""

# 训练工作流配置
TRAINING_WORKFLOW = {
    'training_stages': {
        'stage1': {
            'name': '序列识别预训练',
            'objective': '训练模型识别基本的SBS序列模式',
            'duration': '20 epochs',
            'validation_frequency': 2,
            'save_frequency': 5
        },
        'stage2': {
            'name': '市场结构分析',
            'objective': '增强模型对市场结构的理解',
            'duration': '30 epochs',
            'validation_frequency': 3,
            'save_frequency': 5
        },
        'stage3': {
            'name': '交易信号生成',
            'objective': '优化模型的交易信号生成能力',
            'duration': '50 epochs',
            'validation_frequency': 5,
            'save_frequency': 5
        }
    },
    'checkpointing': {
        'enabled': True,
        'save_best_only': True,
        'metric': 'val_loss',
        'mode': 'min'
    },
    'monitoring': {
        'tensorboard': True,
        'save_plots': True,
        'metrics_history': True
    }
}

# 自监督学习配置
SELF_SUPERVISED_CONFIG = {
    'model': {
        'image_size': (224, 224),
        'sequence_length': 100,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 6
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'warmup_steps': 1000
    },
    'data': {
        'window_size': 100,
        'stride': 20,
        'min_sequence_length': 50,
        'max_sequence_length': 200
    },
    'augmentation': {
        'enabled': True,
        'time_scaling': True,
        'price_scaling': True,
        'noise_injection': True
    },
    'validation': {
        'metrics': ['accuracy', 'precision', 'recall', 'f1'],
        'min_confidence': 0.8,
        'confusion_matrix': True,
        'save_predictions': True
    }
}

# 系统配置
SYSTEM_CONFIG = {
    'run_mode': {
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug': bool(os.getenv('DEBUG', False)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'use_gpu': True,
        'num_workers': int(os.getenv('NUM_WORKERS', 4)),
        'profile_code': False,
        'memory_tracking': True
    },
    'monitor': {
        'system': {
            'cpu_threshold': 90.0,
            'memory_threshold': 85.0,
            'gpu_memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'check_interval': 60,
            'history_size': 1000,
            'alert_channels': ['discord', 'email']
        },
        'trading': {
            'max_daily_trades': 10,
            'max_concurrent_positions': 1,
            'max_daily_drawdown': 0.05,
            'alert_thresholds': {
                'drawdown': 0.03,
                'profit': 0.05,
                'loss': 0.02
            }
        }
    },
    'discord': {
        'webhooks': {
            'monitor': os.getenv('DISCORD_WEBHOOK_MONITOR'),
            'signal': os.getenv('DISCORD_WEBHOOK_SIGNAL'),
            'debug': os.getenv('DISCORD_WEBHOOK_DEBUG')
        },
        'bot': {
            'username': 'SBS Trading Bot',
            'avatar_url': os.getenv('DISCORD_BOT_AVATAR'),
            'embed_color': 0x00ff00
        }
    },
    'logging': {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/sbs_system.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
}

# 数据处理配置
DATA_CONFIG = {
    'preprocessing': {
        'image': {
            'target_size': (224, 224),
            'quality_threshold': 0.8,
            'brightness_range': (0.2, 0.8),
            'contrast_range': (0.4, 0.8),
            'noise_threshold': 0.1,
            'cache_size': 1000,
            'augmentation': {
                'enabled': True,
                'rotation_range': 5,
                'zoom_range': 0.1,
                'brightness_range': (0.9, 1.1),
                'contrast_range': (0.9, 1.1)
            }
        },
        'sequence': {
            'min_length': 100,
            'max_length': 200,
            'validation_window': 100,
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'indicators': {
                'ma': [20, 50, 200],
                'rsi': [14],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            }
        }
    },
    'collection': {
        'screenshot': {
            'interval': 60,
            'quality': 95,
            'max_retries': 3,
            'timeout': 30
        },
        'tradingview': {
            'charts': {
                'NQ1!': {
                    'timeframes': ['1m', '5m', '15m'],
                    'indicators': ['MA', 'Volume', 'RSI']
                }
            }
        }
    }
}

# 交易配置
TRADING_CONFIG = {
    'signal': {
        'min_confidence': 0.75,
        'max_risk_ratio': 0.02,
        'min_reward_ratio': 2.0,
        'entry_zone_size': 0.001,
        'max_stop_distance': 0.02
    },
    'risk_management': {
        'position_sizing': {
            'method': 'fixed_risk',
            'risk_per_trade': 0.01,
            'max_position_size': 0.1
        },
        'stop_loss': {
            'type': 'adaptive',
            'atr_multiplier': 2.0,
            'max_loss': 0.02
        },
        'take_profit': {
            'type': 'multi_target',
            'targets': [2.0, 3.0, 5.0],
            'position_scale': [0.5, 0.3, 0.2]
        }
    }
}

# 在CHART_CONFIG前添加
CHART_SYMBOLS = ['NQ1!', 'ES1!', 'YM1!']  # 交易品种

# 更新CHART_CONFIG
CHART_CONFIG = {
    'symbols': CHART_SYMBOLS,  # 使用常量
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],  # 时间周期
    'indicators': {
        'default': ['MA', 'Volume', 'RSI'],
        'advanced': ['MACD', 'BB', 'ATR']
    },
    'layout': {
        'theme': 'dark',
        'show_grid': True,
        'show_volume': True,
        'chart_type': 'candles'
    }
}

# 移动平均线配置
MA_SETTINGS = {
    'short_term': {
        'periods': [5, 10, 20],
        'type': 'EMA'
    },
    'medium_term': {
        'periods': [50, 100],
        'type': 'SMA'
    },
    'long_term': {
        'periods': [200],
        'type': 'SMA'
    }
}

# SBS标注配置
SBS_ANNOTATION_CONFIG = {
    'points': {
        'breakout': {
            'name': '突破点',
            'color': (255, 0, 0),  # 红色
            'marker_size': 10
        },
        'point1': {
            'name': '点1(首次回调)',
            'color': (0, 255, 0),  # 绿色
            'marker_size': 10
        },
        'point2': {
            'name': '点2(目标位)',
            'color': (0, 0, 255),  # 蓝色
            'marker_size': 10
        },
        'point3': {
            'name': '点3(流动性获取)',
            'color': (255, 165, 0),  # 橙色
            'marker_size': 10
        },
        'point4': {
            'name': '点4(确认点)',
            'color': (128, 0, 128),  # 紫色
            'marker_size': 10
        }
    },
    'font': {
        'size': 12,
        'color': (255, 255, 255),  # 白色
        'thickness': 2
    },
    'line': {
        'color': (255, 255, 0),  # 黄色
        'thickness': 2
    }
}

# 模型配置
MODEL_CONFIG = {
    'base_model': 'models/llava-sbs',
    'device': 'cuda',
    'max_new_tokens': 1000,
    'temperature': 0.7,
    'top_p': 0.9
}

# 图像处理配置
IMAGE_CONFIG = {
    'max_size': 1024,
    'quality': 95,
    'format': 'PNG',
    'dpi': 300
}

# 分析提示模板
ANALYSIS_PROMPT = """
你是一个专业的金融图表分析专家，专注于识别和分析SBS(Sequence Based Signal)交易序列。
请分析这张图表中的以下要素：

1. SBS序列的完整性和有效性：
   - 突破的清晰度和有效性
   - 回调的规范性（0.382-0.618）
   - 确认阶段的完整性
   - 整体序列的时间结构

2. 关键点位标识：
   - 突破点：价格突破前期高点/低点的位置
   - 点1：突破后的第一次回调位置
   - 点2：回调创造的最高/最低点
   - 点3：流动性获取位置
   - 点4：确认点位置

3. 趋势分析：
   - SMA20和SMA200趋势方向
   - 市场结构（Higher Highs/Lower Lows）
   - 整体趋势评估

4. 交易信号：
   - 方向（多/空）
   - 入场区域
   - 止损位置
   - 目标位置

5. 风险评估：
   - 风险等级（低/中/高）
   - 主要风险点
   - 成功概率评估

请按以下格式输出分析结果：

序列评估：
- 有效性：[是/否]
- 完整度：[0-100%]
- 可信度：[0-100%]

关键点位：
- 突破点：[价格水平]
- Point 1：[价格水平]
- Point 2：[价格水平]
- Point 3：[价格水平]
- Point 4：[价格水平]

交易信号：
- 方向：[多/空]
- 入场区域：[价格范围]
- 止损位：[价格水平]
- 目标位：[价格水平]

趋势辅助分析：
- SMA20趋势：[上升/下降/盘整]
- SMA200趋势：[上升/下降/盘整]
- 整体趋势评估：[描述]

风险评估：
- 风险等级：[低/中/高]
- 主要风险点：[描述]
- 成功概率：[0-100%]

注意事项：
1. 所有价格水平必须精确到小数点后4位
2. 确保所有关键点位的时间顺序正确
3. 对于不完整或不确定的形态，请明确指出原因
"""

def load_config() -> Dict[str, Any]:
    """加载系统配置"""
    return {
        'paths': PATHS,
        'model': {
            'path': LLAVA_MODEL_PATH,
            'prompt_template': LLAVA_PROMPT_TEMPLATE,
            'sbs_prompt': SBS_PROMPT  # 添加SBS专用prompt配置
        },
        'training': TRAINING_WORKFLOW,
        'self_supervised': SELF_SUPERVISED_CONFIG,
        'system': SYSTEM_CONFIG,
        'data': DATA_CONFIG,
        'trading': TRADING_CONFIG,
        'chart': CHART_CONFIG,  # 添加图表配置
        'ma': MA_SETTINGS,  # 添加移动平均线配置
        'screenshot_dir': SCREENSHOT_DIR,
        'clip_model_path': CLIP_MODEL_PATH,
        'sbs_annotation': SBS_ANNOTATION_CONFIG,
        'model_config': MODEL_CONFIG,
        'image_config': IMAGE_CONFIG,
        'analysis_prompt': ANALYSIS_PROMPT
    }
"""
主程序入口
"""
import os
import logging
from pathlib import Path
import json

from .bot.discord_bot import run_bot
from .utils.logger import setup_logger

logger = setup_logger('main')

def load_config() -> dict:
    """加载配置文件
    
    Returns:
        dict: 配置字典
    """
    try:
        config_path = Path('config/config.json')
        if not config_path.exists():
            # 创建默认配置
            config = {
                'system': {
                    'device': 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu',
                    'num_workers': int(os.getenv('NUM_WORKERS', '4')),
                    'debug': os.getenv('DEBUG', 'false').lower() == 'true'
                },
                'model': {
                    'base_model': os.getenv('MODEL_PATH', 'models/llava-sbs'),
                    'vision_model': os.getenv('VISION_MODEL_PATH', 'openai/clip-vit-large-patch14-336'),
                    'max_length': int(os.getenv('MAX_LENGTH', '4096')),
                    'batch_size': int(os.getenv('BATCH_SIZE', '4'))
                },
                'discord': {
                    'token': os.getenv('DISCORD_BOT_TOKEN'),
                    'client_id': os.getenv('DISCORD_CLIENT_ID'),
                    'signal_webhook': os.getenv('DISCORD_WEBHOOK_SIGNAL'),
                    'monitor_webhook': os.getenv('DISCORD_WEBHOOK_MONITOR'),
                    'upload_webhook': os.getenv('DISCORD_WEBHOOK_UPLOAD'),
                    'avatar_url': os.getenv('DISCORD_BOT_AVATAR')
                }
            }
            
            # 保存配置
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
        else:
            # 加载现有配置
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        return config
        
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise

def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        
        # 检查必要的环境变量
        if not config['discord']['token']:
            raise ValueError("未设置DISCORD_BOT_TOKEN")
            
        # 运行Discord Bot
        run_bot(config['discord']['token'], config)
        
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        raise

if __name__ == "__main__":
    main() 
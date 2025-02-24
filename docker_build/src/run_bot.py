import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot.discord_bot import run_bot
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('bot_runner')

def setup_proxy():
    """设置代理配置"""
    try:
        # 设置代理环境变量
        os.environ['http_proxy'] = 'http://127.0.0.1:7897'
        os.environ['https_proxy'] = 'http://127.0.0.1:7897'
        os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'
        
        # 验证代理连接
        import requests
        response = requests.get('https://api.discord.com', timeout=5)
        if response.status_code == 200:
            logger.info("代理配置成功")
        else:
            logger.warning("代理连接可能存在问题")
    except Exception as e:
        logger.error(f"代理设置失败: {e}")

def load_config(config_path: str) -> dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def main():
    """主函数"""
    try:
        # 设置代理
        setup_proxy()
        
        # 加载环境变量
        load_dotenv()
        
        # 获取Discord Token
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token:
            raise ValueError("未设置DISCORD_BOT_TOKEN环境变量")
            
        # 加载配置
        config_path = Path('config/bot_config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        config = load_config(str(config_path))
        
        # 运行Bot
        logger.info("正在启动Discord Bot...")
        run_bot(token, config)
        
    except Exception as e:
        logger.error(f"Bot启动失败: {e}")
        raise

if __name__ == '__main__':
    main() 
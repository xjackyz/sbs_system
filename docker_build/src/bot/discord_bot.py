import discord
from discord import Webhook
import aiohttp
import logging
from typing import Optional, Dict
from pathlib import Path
import os

from ..model.sbs_analyzer import SBSAnalyzer
from ..notification.discord_notifier import DiscordNotifier
from ..utils.logger import setup_logger

logger = setup_logger('discord_bot')

class SBSDiscordBot(discord.Client):
    def __init__(self, config: Dict):
        """初始化Discord机器人
        
        Args:
            config: 配置字典
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.config = config
        self.analyzer = None
        self.notifier = None
        
        # 创建临时目录用于存储图片
        self.temp_dir = Path('temp')
        self.temp_dir.mkdir(exist_ok=True)
        
    async def setup_hook(self):
        """Bot启动时的设置"""
        try:
            # 初始化分析器
            self.analyzer = SBSAnalyzer(
                base_model=self.config['model']['base_model'],
                device=self.config['system']['device']
            )
            
            # 初始化通知器
            self.notifier = DiscordNotifier(self.config['discord'])
            
            logger.info("Bot设置完成")
            
            # 发送启动通知
            await self.notifier.send_monitor_message({
                'type': 'status',
                'message': f'Bot已启动: {self.user.name}'
            })
            
        except Exception as e:
            logger.error(f"Bot设置失败: {e}")
            raise
            
    async def on_ready(self):
        """Bot就绪事件处理"""
        logger.info(f"Bot已登录: {self.user.name}")
        
    async def on_message(self, message: discord.Message):
        """消息事件处理"""
        try:
            # 检查是否是上传频道的消息
            if message.channel.id != self.config['discord']['upload_channel_id']:
                return
                
            # 检查是否包含图片
            if not message.attachments:
                return
                
            # 检查图片格式
            attachment = message.attachments[0]
            if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return
                
            # 下载图片
            image_path = await self.download_image(attachment)
            if not image_path:
                logger.error("图片下载失败")
                return
                
            try:
                # 分析图表
                analysis_result = await self.analyze_chart(image_path)
                
                # 发送分析结果到信号频道
                await self.notifier.send_signal(analysis_result)
                
            except Exception as e:
                logger.error(f"分析过程出错: {e}")
                await self.notifier.send_monitor_message({
                    'type': 'error',
                    'message': f'分析失败: {str(e)}',
                    'user': str(message.author)
                })
                
            finally:
                # 清理临时文件
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
        except Exception as e:
            logger.error(f"消息处理错误: {e}")
            
    async def download_image(self, attachment) -> Optional[str]:
        """下载图片
        
        Args:
            attachment: Discord附件对象
            
        Returns:
            str: 临时文件路径，如果下载失败则返回None
        """
        try:
            # 生成临时文件路径
            temp_path = self.temp_dir / f"temp_{attachment.filename}"
            
            # 下载文件
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as response:
                    if response.status != 200:
                        return None
                        
                    data = await response.read()
                    with open(temp_path, 'wb') as f:
                        f.write(data)
                        
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None
            
    async def analyze_chart(self, image_path: str) -> Dict:
        """分析图表
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 使用SBS分析器分析图表
            result = self.analyzer.process_image(image_path)
            
            # 构建结果字典
            analysis_result = {
                'symbol': 'Unknown',  # TODO: 从图片中识别交易对
                'timeframe': '1h',    # TODO: 从图片中识别时间周期
                'signal_type': result.get('step', 'Unknown'),
                'confidence_score': result.get('confidence', 0),
                'metadata': {
                    'pattern_type': 'SBS',
                    'market_condition': result.get('market_condition', 'Unknown')
                }
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"图表分析失败: {e}")
            raise
            
def run_bot(token: str, config: Dict):
    """运行Discord Bot
    
    Args:
        token: Discord Bot Token
        config: 配置字典
    """
    bot = SBSDiscordBot(config)
    bot.run(token) 
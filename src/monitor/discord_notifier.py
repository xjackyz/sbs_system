"""
Discord通知器模块
"""
from typing import Dict, Optional, Any
import discord
import aiohttp
import asyncio
from datetime import datetime
import json
import logging

from src.utils.logger import setup_logger

logger = setup_logger('discord_notifier')

class DiscordNotifier:
    """Discord通知器类"""
    def __init__(self, webhook_urls: Dict[str, str], username: str = "SBS Trading Bot"):
        self.webhook_urls = webhook_urls
        self.username = username
        self.webhooks = {}
        self.session = None
        
    async def setup(self):
        """设置异步会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            # 创建webhook对象
            for webhook_type, url in self.webhook_urls.items():
                self.webhooks[webhook_type] = discord.Webhook.from_url(
                    url,
                    session=self.session
                )
            
    async def cleanup(self):
        """清理资源"""
        if self.session is not None:
            await self.session.close()
            self.session = None
            self.webhooks = {}
            
    async def send_message(self, message: str, webhook_type: str = 'monitor',
                        embed: Optional[discord.Embed] = None):
        """发送消息"""
        try:
            if self.session is None:
                await self.setup()
                
            webhook = self.webhooks.get(webhook_type)
            if not webhook:
                logger.error(f"未找到webhook类型: {webhook_type}")
                return
                
            await webhook.send(
                content=message,
                username=self.username,
                embed=embed
            )
            logger.info(f"消息发送成功: {webhook_type}")
                    
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            
    async def send_system_alert(self, alert_data: Dict[str, Any]):
        """发送系统告警"""
        try:
            embed = discord.Embed(
                title="系统告警",
                description=alert_data.get('message', '无告警信息'),
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            
            # 添加告警详情
            for key, value in alert_data.items():
                if key != 'message':
                    embed.add_field(name=key, value=str(value), inline=True)
                    
            await self.send_message("", 'monitor', embed)
            
        except Exception as e:
            logger.error(f"发送系统告警失败: {e}")
            
    async def send_trading_signal(self, signal_data: Dict[str, Any]):
        """发送交易信号"""
        try:
            # 设置嵌入消息颜色
            color = discord.Color.green() if signal_data.get('direction') == 'buy' else discord.Color.red()
            
            embed = discord.Embed(
                title=f"交易信号 - {signal_data.get('symbol')}",
                description=signal_data.get('description', '无信号描述'),
                color=color,
                timestamp=datetime.now()
            )
            
            # 添加信号详情
            embed.add_field(name="方向", value=signal_data.get('direction', 'unknown'), inline=True)
            embed.add_field(name="入场价", value=signal_data.get('entry_price', 'N/A'), inline=True)
            embed.add_field(name="止损", value=signal_data.get('stop_loss', 'N/A'), inline=True)
            embed.add_field(name="目标", value=signal_data.get('take_profit', 'N/A'), inline=True)
            embed.add_field(name="置信度", value=f"{signal_data.get('confidence', 0):.2%}", inline=True)
            
            await self.send_message("", 'signal', embed)
            
        except Exception as e:
            logger.error(f"发送交易信号失败: {e}")
            
    async def send_debug_info(self, debug_data: Dict[str, Any]):
        """发送调试信息"""
        try:
            embed = discord.Embed(
                title="调试信息",
                description=debug_data.get('message', '无调试信息'),
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            # 添加调试详情
            for key, value in debug_data.items():
                if key != 'message':
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, indent=2)
                    embed.add_field(name=key, value=str(value), inline=False)
                    
            await self.send_message("", 'debug', embed)
            
        except Exception as e:
            logger.error(f"发送调试信息失败: {e}")
            
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.cleanup() 
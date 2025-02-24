"""Discord 处理模块"""
import discord
from discord import File, Webhook
import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import io
import json
from datetime import datetime

class DiscordHandler:
    def __init__(self, config: Dict[str, Any]):
        """初始化Discord处理器
        
        Args:
            config: Discord配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.webhooks = {
            'monitor': config['webhooks'].get('monitor'),
            'signal': config['webhooks'].get('signal'),
            'debug': config['webhooks'].get('debug')
        }
        self.bot_avatar = config.get('bot_avatar')
        
    async def send_image_analysis(self,
                                image_path: str,
                                analysis_result: Dict[str, Any],
                                webhook_type: str = 'signal') -> bool:
        """发送图片分析结果
        
        Args:
            image_path: 图片路径
            analysis_result: 分析结果
            webhook_type: webhook类型
        
        Returns:
            bool: 是否发送成功
        """
        try:
            webhook_url = self.webhooks.get(webhook_type)
            if not webhook_url:
                self.logger.error(f"未找到{webhook_type}类型的webhook配置")
                return False
                
            # 准备图片文件
            image_file = Path(image_path)
            if not image_file.exists():
                self.logger.error(f"图片文件不存在: {image_path}")
                return False
                
            # 构建嵌入消息
            embed = self._create_analysis_embed(analysis_result)
            
            async with aiohttp.ClientSession() as session:
                # 创建webhook
                webhook = Webhook.from_url(webhook_url, session=session)
                
                # 发送消息
                with open(image_path, 'rb') as f:
                    file = File(f, filename=image_file.name)
                    await webhook.send(
                        embed=embed,
                        file=file,
                        avatar_url=self.bot_avatar
                    )
                    
            self.logger.info(f"成功发送分析结果到Discord: {image_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送Discord消息失败: {str(e)}")
            return False
            
    async def send_alert(self,
                        alert_data: Dict[str, Any],
                        webhook_type: str = 'signal') -> bool:
        """发送交易提醒
        
        Args:
            alert_data: 提醒数据
            webhook_type: webhook类型
            
        Returns:
            bool: 是否发送成功
        """
        try:
            webhook_url = self.webhooks.get(webhook_type)
            if not webhook_url:
                self.logger.error(f"未找到{webhook_type}类型的webhook配置")
                return False
                
            # 构建嵌入消息
            embed = self._create_alert_embed(alert_data)
            
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(webhook_url, session=session)
                await webhook.send(
                    embed=embed,
                    avatar_url=self.bot_avatar
                )
                
            self.logger.info(f"成功发送交易提醒到Discord")
            return True
            
        except Exception as e:
            self.logger.error(f"发送Discord提醒失败: {str(e)}")
            return False
            
    def _create_analysis_embed(self, analysis_result: Dict[str, Any]) -> discord.Embed:
        """创建分析结果的嵌入消息
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            discord.Embed: Discord嵌入消息
        """
        # 获取分析结果的关键信息
        sequence_type = analysis_result.get('sequence_type', 'Unknown')
        confidence = analysis_result.get('confidence', 0.0)
        timestamp = analysis_result.get('timestamp', datetime.now().isoformat())
        
        # 创建嵌入消息
        embed = discord.Embed(
            title="SBS序列分析结果",
            description=f"检测到 {sequence_type} 序列",
            color=self._get_sequence_color(sequence_type),
            timestamp=datetime.fromisoformat(timestamp)
        )
        
        # 添加字段
        embed.add_field(
            name="序列类型",
            value=sequence_type,
            inline=True
        )
        embed.add_field(
            name="置信度",
            value=f"{confidence:.2%}",
            inline=True
        )
        
        # 添加其他分析信息
        if 'details' in analysis_result:
            embed.add_field(
                name="详细分析",
                value=analysis_result['details'],
                inline=False
            )
            
        return embed
        
    def _create_alert_embed(self, alert_data: Dict[str, Any]) -> discord.Embed:
        """创建交易提醒的嵌入消息
        
        Args:
            alert_data: 提醒数据
            
        Returns:
            discord.Embed: Discord嵌入消息
        """
        # 获取提醒数据的关键信息
        symbol = alert_data.get('symbol', 'Unknown')
        action = alert_data.get('action', 'Unknown')
        price = alert_data.get('price', 0.0)
        
        # 创建嵌入消息
        embed = discord.Embed(
            title="交易信号提醒",
            description=f"{symbol} - {action}",
            color=self._get_action_color(action),
            timestamp=datetime.now()
        )
        
        # 添加字段
        embed.add_field(
            name="交易对",
            value=symbol,
            inline=True
        )
        embed.add_field(
            name="操作",
            value=action,
            inline=True
        )
        embed.add_field(
            name="价格",
            value=f"${price:,.2f}",
            inline=True
        )
        
        # 添加其他信号信息
        if 'stop_loss' in alert_data:
            embed.add_field(
                name="止损",
                value=f"${alert_data['stop_loss']:,.2f}",
                inline=True
            )
        if 'take_profit' in alert_data:
            embed.add_field(
                name="止盈",
                value=f"${alert_data['take_profit']:,.2f}",
                inline=True
            )
            
        return embed
        
    def _get_sequence_color(self, sequence_type: str) -> int:
        """获取序列类型对应的颜色
        
        Args:
            sequence_type: 序列类型
            
        Returns:
            int: Discord颜色代码
        """
        color_map = {
            'SETUP': 0x00ff00,  # 绿色
            'ENTRY': 0x0000ff,  # 蓝色
            'EXIT': 0xff0000,   # 红色
            'ALERT': 0xffff00   # 黄色
        }
        return color_map.get(sequence_type, 0x808080)  # 默认为灰色
        
    @staticmethod
    def _get_action_color(action: str) -> int:
        """获取交易动作对应的颜色
        
        Args:
            action: 交易动作
            
        Returns:
            int: Discord颜色代码
        """
        colors = {
            'BUY': 0x2ecc71,     # 绿色
            'SELL': 0xe74c3c,    # 红色
            'HOLD': 0xf1c40f,    # 黄色
            'WATCH': 0x3498db    # 蓝色
        }
        return colors.get(action.upper(), 0x7f8c8d)
        
    async def test_connection(self) -> bool:
        """测试Discord连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            webhook_url = self.webhooks.get('debug')
            if not webhook_url:
                self.logger.error("未找到debug webhook配置")
                return False
                
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(webhook_url, session=session)
                await webhook.send(
                    content="连接测试成功",
                    avatar_url=self.bot_avatar
                )
            return True
            
        except Exception as e:
            self.logger.error(f"Discord连接测试失败: {str(e)}")
            return False 
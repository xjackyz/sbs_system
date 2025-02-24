"""
Discord通知模块
"""
import discord
from discord import Webhook, Embed
import aiohttp
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import logging
import os
from PIL import Image
import io

from ..utils.logger import setup_logger

logger = setup_logger('discord_notifier')

@dataclass
class DiscordConfig:
    """Discord配置"""
    webhooks: Dict[str, str]  # 不同类型通知的webhook URLs
    username: str = "SBS Trading Bot"  # 机器人用户名
    avatar_url: Optional[str] = None  # 机器人头像URL
    embed_color: int = 0x00ff00  # 嵌入消息颜色
    max_retries: int = 3  # 最大重试次数
    retry_delay: int = 5  # 重试延迟（秒）

class DiscordNotifier:
    def __init__(self, discord_config: Optional[DiscordConfig] = None):
        """
        初始化Discord通知器
        
        Args:
            discord_config: Discord配置
        """
        self.config = discord_config or self._load_default_config()
        self.message_templates = self.load_message_templates()
        
    def _load_default_config(self) -> DiscordConfig:
        """加载默认配置"""
        return DiscordConfig(
            webhooks={
                'signal': os.getenv('DISCORD_WEBHOOK_SIGNAL'),
                'upload': os.getenv('DISCORD_WEBHOOK_UPLOAD'),
                'monitor': os.getenv('DISCORD_WEBHOOK_MONITOR')
            }
        )
        
    def load_message_templates(self) -> Dict[str, str]:
        """加载消息模板"""
        return {
            'signal': """
📊 交易信号 - {symbol} ({timeframe})
------------------
📈 信号类型: {signal_type}
⬆️ 方向: {direction}
💰 入场价格: {entry_price}
🛑 止损价格: {stop_loss}
🎯 目标价格: {take_profit}
⚡️ 置信度: {confidence:.2%}

🔍 模式类型: {pattern_type}
📊 市场状况: {market_condition}
⏰ 时间: {timestamp}
            """.strip(),
            
            'monitor': """
🔄 系统状态
------------------
💻 状态: {status}
📈 处理序列: {sequence_count}
⚡️ 信号数量: {signal_count}
🎯 成功率: {success_rate:.2%}
⏰ 时间: {timestamp}
            """.strip(),
            
            'error': """
⚠️ 错误通知
------------------
类型: {type}
消息: {message}
用户: {user}
⏰ 时间: {timestamp}
            """.strip(),
            
            'status': """
ℹ️ 状态通知
------------------
{message}
⏰ 时间: {timestamp}
            """.strip()
        }
        
    async def send_signal(self, channel, analysis_result: Dict):
        """发送交易信号
        
        Args:
            channel: Discord频道
            analysis_result: 分析结果
        """
        try:
            if analysis_result.get('success', False):
                # 创建嵌入消息
                embed = self._create_analysis_embed(analysis_result)
                
                if 'annotated_image' in analysis_result:
                    # 发送标注后的图片
                    await channel.send(
                        file=discord.File(analysis_result['annotated_image']),
                        embed=embed
                    )
                else:
                    # 发送分析结果但没有图片
                    await channel.send(embed=embed)
            else:
                # 发送错误消息
                error_embed = discord.Embed(
                    title="分析失败",
                    description=f"错误信息: {analysis_result.get('error', '未知错误')}",
                    color=0xff0000,
                    timestamp=datetime.now()
                )
                await channel.send(embed=error_embed)
                
        except Exception as e:
            logger.error(f"发送信号失败: {str(e)}")
            # 尝试发送简单的错误消息
            try:
                await channel.send(f"发送分析结果时出错: {str(e)}")
            except:
                logger.error("发送错误消息也失败了")
            
    def _create_analysis_embed(self, analysis_result: Dict) -> discord.Embed:
        """创建分析结果的嵌入消息
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            discord.Embed: 嵌入消息
        """
        try:
            # 创建基础嵌入消息
            embed = discord.Embed(
                title="SBS交易信号分析",
                color=self._get_signal_color(analysis_result),
                timestamp=datetime.now()
            )
            
            # 添加序列评估
            if 'sequence_evaluation' in analysis_result:
                eval_data = analysis_result['sequence_evaluation']
                embed.add_field(
                    name="📊 序列评估",
                    value=f"有效性: {'✅' if eval_data.get('validity') == '是' else '❌'}\n"
                          f"完整度: {eval_data.get('completeness', 0)}%\n"
                          f"可信度: {eval_data.get('confidence', 0)}%",
                    inline=False
                )
            
            # 添加交易信号
            if 'trading_signal' in analysis_result:
                signal_data = analysis_result['trading_signal']
                entry_zone = signal_data.get('entry_zone', {})
                embed.add_field(
                    name="📈 交易信号",
                    value=f"方向: {self._get_direction_emoji(signal_data.get('direction'))} {signal_data.get('direction', '未知')}\n"
                          f"入场区域: {entry_zone.get('min', '未知')}-{entry_zone.get('max', '未知')}\n"
                          f"止损位: {signal_data.get('stop_loss', '未知')}\n"
                          f"目标位: {signal_data.get('target', '未知')}",
                    inline=False
                )
            
            # 添加关键点位
            if 'key_points' in analysis_result:
                points_data = analysis_result['key_points']
                embed.add_field(
                    name="🎯 关键点位",
                    value=f"突破点: {points_data.get('breakout', '未知')}\n"
                          f"Point 1: {points_data.get('point1', '未知')}\n"
                          f"Point 2: {points_data.get('point2', '未知')}\n"
                          f"Point 3: {points_data.get('point3', '未知')}\n"
                          f"Point 4: {points_data.get('point4', '未知')}",
                    inline=False
                )
            
            # 添加趋势分析
            if 'trend_analysis' in analysis_result:
                trend_data = analysis_result['trend_analysis']
                embed.add_field(
                    name="📊 趋势分析",
                    value=f"SMA20: {self._get_trend_emoji(trend_data.get('sma20_trend'))} {trend_data.get('sma20_trend', '未知')}\n"
                          f"SMA200: {self._get_trend_emoji(trend_data.get('sma200_trend'))} {trend_data.get('sma200_trend', '未知')}\n"
                          f"整体趋势: {trend_data.get('overall_trend', '未知')}",
                    inline=False
                )
            
            # 添加风险评估
            if 'risk_assessment' in analysis_result:
                risk_data = analysis_result['risk_assessment']
                embed.add_field(
                    name="⚠️ 风险评估",
                    value=f"风险等级: {self._get_risk_emoji(risk_data.get('risk_level'))} {risk_data.get('risk_level', '未知')}\n"
                          f"主要风险: {risk_data.get('main_risks', '未知')}",
                    inline=False
                )
            
            return embed
            
        except Exception as e:
            logger.error(f"创建嵌入消息失败: {e}")
            # 返回一个简单的错误嵌入消息
            return discord.Embed(
                title="分析结果格式化失败",
                description="创建详细分析报告时出错",
                color=0xff0000
            )
            
    def _get_signal_color(self, analysis_result: Dict) -> int:
        """获取信号颜色
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            int: 颜色代码
        """
        if 'trading_signal' not in analysis_result:
            return self.config.embed_color
            
        direction = analysis_result['trading_signal'].get('direction', '').lower()
        if direction == '多':
            return 0x00ff00  # 绿色
        elif direction == '空':
            return 0xff0000  # 红色
        else:
            return 0xffff00  # 黄色
            
    @staticmethod
    def _get_direction_emoji(direction: Optional[str]) -> str:
        """获取方向emoji
        
        Args:
            direction: 方向
            
        Returns:
            str: emoji
        """
        if not direction:
            return '❓'
        direction = direction.lower()
        if direction == '多':
            return '📈'
        elif direction == '空':
            return '📉'
        else:
            return '➡️'
            
    @staticmethod
    def _get_trend_emoji(trend: Optional[str]) -> str:
        """获取趋势emoji
        
        Args:
            trend: 趋势
            
        Returns:
            str: emoji
        """
        if not trend:
            return '❓'
        trend = trend.lower()
        if trend == '上升':
            return '📈'
        elif trend == '下降':
            return '📉'
        else:
            return '➡️'
            
    @staticmethod
    def _get_risk_emoji(risk_level: Optional[str]) -> str:
        """获取风险等级emoji
        
        Args:
            risk_level: 风险等级
            
        Returns:
            str: emoji
        """
        if not risk_level:
            return '❓'
        risk_level = risk_level.lower()
        if risk_level == '低':
            return '🟢'
        elif risk_level == '中':
            return '🟡'
        elif risk_level == '高':
            return '🔴'
        else:
            return '❓'
            
    async def send_monitor_message(self, data: Dict):
        """发送监控消息
        
        Args:
            data: 消息数据
        """
        try:
            # 创建嵌入消息
            embed = discord.Embed(
                description=f"类型: {data.get('type', 'info')}\n"
                           f"消息: {data.get('message', '')}\n"
                           f"错误: {data.get('error', '无')}",
                color=0xff0000 if data.get('type') == 'error' else self.config.embed_color,
                timestamp=datetime.now()
            )
            
            # 发送消息
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(
                    self.config.webhooks['monitor'],
                    session=session
                )
                await webhook.send(
                    username=self.config.username,
                    avatar_url=self.config.avatar_url,
                    embed=embed
                )
            
            logger.info(f"监控消息发送成功: {data.get('type', 'info')}")
            
        except Exception as e:
            logger.error(f"发送监控消息失败: {e}")
            
    async def send_error_message(self, channel, error: str):
        """发送错误消息
        
        Args:
            channel: Discord频道
            error: 错误信息
        """
        try:
            embed = discord.Embed(
                title="❌ 错误",
                description=error,
                color=0xff0000,
                timestamp=datetime.now()
            )
            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"发送错误消息失败: {e}")
            try:
                await channel.send(f"错误: {error}")
            except:
                logger.error("发送简单错误消息也失败了")
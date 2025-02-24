import asyncio
import aiohttp
import logging
import discord
from discord import Webhook, File, Embed
import os
from datetime import datetime
import sys
import requests
import socket
import urllib3
import ssl
import json
from discord.ext import commands
import warnings
from src.model.llava_analyzer import LLaVAAnalyzer
from src.config import Config
from src.utils.trading_alerts import TradingAlertManager
from aiohttp_socks import ProxyConnector

# 忽略不安全的HTTPS警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# 配置详细的日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discord_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# 启用 discord.py 的详细日志
discord_logger = logging.getLogger('discord')
discord_logger.setLevel(logging.DEBUG)
discord_handler = logging.FileHandler('discord_debug.log')
discord_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
discord_logger.addHandler(discord_handler)

# Discord配置
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 从环境变量获取
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")  # 从环境变量获取
PERMISSIONS = int(os.getenv("DISCORD_PERMISSIONS", "1689934340028480"))  # 从环境变量获取，默认值为示例权限

# Webhook URLs
MONITOR_WEBHOOK = "https://discord.com/api/webhooks/1343455788697518133/eMO_2hFoerAliK6eBct00rD5U8k-IXGEeD6-Jg0k30_54A7Uchi-IPdbL3LHPYUnPAkA"
SIGNAL_WEBHOOK = "https://discord.com/api/webhooks/1343455498187571312/NNHdDTLdTE1Lg5PVKojiMM4CT2_8lNcdnpGlBdIHTTQHAfQ-BeZFAHJtlaNErPZkXBDA"
UPLOAD_WEBHOOK = "https://discord.com/api/webhooks/1343455502352388179/G_Vkp50OqNErkWgXAMKlKEECBQ5qOj-g3lkArCiofkdnUN9456uANEHEOEoY_qaFJx-4"

# 设置代理
PROXY_URL = "http://127.0.0.1:7897"  # Clash 的混合端口
PROXY_CONFIG = {
    'http': PROXY_URL,
    'https': PROXY_URL
}

# 设置环境变量
os.environ['ALL_PROXY'] = PROXY_URL 
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 配置urllib3代理
if PROXY_URL:
    proxy_manager = urllib3.ProxyManager(
        PROXY_URL,
        num_pools=10,
        maxsize=10,
        retries=urllib3.Retry(3),
        timeout=urllib3.Timeout(connect=30, read=60)
    )
else:
    proxy_manager = urllib3.PoolManager(
        num_pools=10,
        maxsize=10,
        retries=urllib3.Retry(3),
        timeout=urllib3.Timeout(connect=30, read=60)
    )

# 生成Bot邀请链接
BOT_INVITE_LINK = f"https://discord.com/api/oauth2/authorize?client_id={CLIENT_ID}&permissions={PERMISSIONS}&scope=bot"

def test_discord_connection():
    try:
        logger.info("开始测试Discord连接...")
        
        # 获取当前代理设置
        current_proxy = os.environ.get('ALL_PROXY', 'none')
        logger.info(f"当前使用的代理: {current_proxy}")
        
        # DNS解析测试
        try:
            ip = socket.gethostbyname('discord.com')
            logger.info(f"DNS解析成功: discord.com -> {ip}")
        except socket.gaierror as e:
            logger.error(f"DNS解析失败: {e}")
            return False
            
        # HTTPS/API测试
        try:
            logger.info("测试Discord API连接...")
            session = requests.Session()
            session.trust_env = True
            session.verify = False
            
            response = session.get(
                'https://discord.com/api/v10/gateway',
                proxies=PROXY_CONFIG,
                timeout=30
            )
            
            logger.info(f"API响应状态码: {response.status_code}")
            logger.info(f"API响应内容: {response.text}")
            
            if response.status_code == 200:
                logger.info("Discord API连接测试成功")
                return True
            else:
                logger.error(f"API请求失败: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求异常: {e}")
            return False
            
    except Exception as e:
        logger.error(f"连接测试过程中发生错误: {e}")
        return False

class TestBot(commands.Bot):
    def __init__(self):
        # 设置所有需要的意图
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)
        
        # 设置重连参数
        self.max_retries = 5
        self.retry_delay = 5
        self.current_retry = 0
        
        # 初始化LLaVA分析器
        try:
            logger.info("正在初始化LLaVA分析器...")
            config = Config(
                model_path="models/llava-sbs",
                device="cuda"
            )
            self.analyzer = LLaVAAnalyzer(config)
        except Exception as e:
            logger.error(f"初始化LLaVA分析器失败: {e}")
            self.analyzer = None
        
        # 初始化交易提醒管理器
        self.alert_manager = TradingAlertManager()
        
    async def setup_hook(self):
        logger.info("开始设置Bot...")
        while self.current_retry < self.max_retries:
            try:
                # 配置代理
                connector = aiohttp.TCPConnector(
                    ssl=True,
                    force_close=False,
                    enable_cleanup_closed=True,
                    verify_ssl=True,
                    ttl_dns_cache=300,
                    limit=100,
                    keepalive_timeout=30
                )
                
                # 创建session
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=60),
                    trust_env=True  # 使用系统代理设置
                )
                
                # 替换discord.py的默认session
                if hasattr(self.http, '_HTTPClient__session'):
                    await self.http._HTTPClient__session.close()
                    self.http._HTTPClient__session = self.session
                
                # 添加更多的事件处理
                self.event(self.on_socket_raw_receive)
                self.event(self.on_socket_raw_send)
                
                # 测试Webhook连接
                logger.info("测试Webhook连接...")
                async with self.session as session:
                    for name, url in {
                        'MONITOR': MONITOR_WEBHOOK,
                        'SIGNAL': SIGNAL_WEBHOOK,
                        'UPLOAD': UPLOAD_WEBHOOK
                    }.items():
                        try:
                            async with session.get(url, ssl=True) as response:
                                logger.info(f"{name} Webhook 测试结果: {response.status}")
                        except Exception as e:
                            logger.error(f"{name} Webhook 测试失败: {e}")
                
                logger.info("Bot设置完成")
                break
                
            except Exception as e:
                self.current_retry += 1
                logger.error(f"Bot设置失败 (尝试 {self.current_retry}/{self.max_retries}): {e}")
                if self.current_retry < self.max_retries:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("已达到最大重试次数，放弃重试")
                    raise
                    
    async def on_socket_raw_receive(self, msg):
        logger.debug(f"Received WebSocket message: {msg[:200]}...")
        
    async def on_socket_raw_send(self, payload):
        logger.debug(f"Sending WebSocket message: {payload[:200]}...")
        
    async def on_ready(self):
        logger.info(f"Bot {self.user} (ID: {self.user.id}) 已准备就绪!")
        logger.info(f"已连接到 {len(self.guilds)} 个服务器:")
        
        for guild in self.guilds:
            logger.info(f"- 服务器: {guild.name} (ID: {guild.id})")
            logger.info(f"  - 成员数: {guild.member_count}")
            logger.info(f"  - 频道数: {len(guild.channels)}")
            
            # 检查Bot权限
            permissions = guild.me.guild_permissions
            logger.info(f"  - Bot权限:")
            logger.info(f"    - 发送消息: {permissions.send_messages}")
            logger.info(f"    - 嵌入链接: {permissions.embed_links}")
            logger.info(f"    - 附加文件: {permissions.attach_files}")
            logger.info(f"    - 读取消息: {permissions.read_messages}")
            
        # 发送测试消息
        try:
            for guild in self.guilds:
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).send_messages:
                        test_embed = discord.Embed(
                            title="Bot 测试消息",
                            description="这是一条测试消息，用于验证Bot的消息发送功能。",
                            color=0x3498db
                        )
                        await channel.send(embed=test_embed)
                        logger.info(f"已发送测试消息到 {guild.name}/{channel.name}")
                        break
        except Exception as e:
            logger.error(f"发送测试消息失败: {e}")
        
    async def on_connect(self):
        """处理连接成功事件"""
        logger.info("Bot已连接到Discord!")
        logger.info(f"Session ID: {self.http.token}")
        self.current_retry = 0  # 重置重试计数器
        
    async def on_disconnect(self):
        """处理断开连接事件"""
        logger.warning("Bot与Discord断开连接!")
        # 记录断开时的状态
        if hasattr(self, 'ws'):
            logger.info(f"WebSocket状态: {self.ws.state}")
        if hasattr(self, 'http'):
            logger.info(f"Session ID: {self.http.token}")
            
    async def on_error(self, event, *args, **kwargs):
        """处理错误事件"""
        logger.error(f"Error in {event}: {sys.exc_info()}")
        logger.exception("详细错误信息:")
        
        # 尝试重新连接
        if event in ['on_ready', 'on_connect']:
            logger.info("尝试重新连接...")
            try:
                await self.close()
                await asyncio.sleep(5)
                await self.start(BOT_TOKEN)
            except Exception as e:
                logger.error(f"重新连接失败: {e}")
                
    async def on_guild_join(self, guild):
        logger.info(f"Joined new guild: {guild.name} (ID: {guild.id})")
        
    async def close(self):
        if hasattr(self, 'session'):
            await self.session.close()
        await super().close()

    async def on_message(self, message):
        if message.author == self.user:
            return
            
        logger.info(f"收到消息: {message.content}")
        logger.info(f"来自: {message.author} in {message.channel}")
        
        if message.attachments:
            logger.info(f"附件数量: {len(message.attachments)}")
            for attachment in message.attachments:
                logger.info(f"处理附件: {attachment.filename}")
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    logger.info(f"收到图片: {attachment.filename}")
                    
                    # 下载图片
                    temp_path = f"temp_{attachment.filename}"
                    await attachment.save(temp_path)
                    
                    try:
                        # 使用LLaVA分析图片
                        logger.info("开始分析图片...")
                        analysis_result = self.analyzer.analyze_image(temp_path)
                        
                        # 生成交易提醒
                        alert = self.alert_manager.generate_alert(
                            step=analysis_result['step'],
                            confidence=analysis_result['confidence'],
                            symbol=analysis_result.get('symbol', 'unknown'),
                            timeframe=analysis_result.get('timeframe', '1h')
                        )
                        
                        # 创建回复消息
                        embed = discord.Embed(
                            title="交易信号分析",
                            description=analysis_result['details'],
                            color=self._get_sequence_color(analysis_result['step'])
                        )
                        
                        embed.add_field(
                            name="交易步骤",
                            value=analysis_result['step'],
                            inline=True
                        )
                        embed.add_field(
                            name="置信度",
                            value=f"{analysis_result['confidence']:.2%}",
                            inline=True
                        )
                        
                        # 发送分析结果
                        await message.channel.send(embed=embed)
                        
                        # 如果置信度高，发送到信号频道
                        if analysis_result['confidence'] > 0.8:
                            signal_webhook = discord.Webhook.from_url(
                                SIGNAL_WEBHOOK,
                                session=self.session
                            )
                            await signal_webhook.send(embed=embed)
                            
                    except Exception as e:
                        logger.error(f"分析图片时出错: {e}")
                        await message.channel.send(f"分析图片时出错: {str(e)}")
                        
                    finally:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
    def _get_sequence_color(self, step: str) -> int:
        """获取序列对应的颜色"""
        colors = {
            'SETUP': 0x3498db,    # 蓝色
            'BUILDUP': 0x2ecc71,  # 绿色
            'STRENGTH': 0xf1c40f, # 黄色
            'BREAKOUT': 0xe74c3c, # 红色
            'SELL': 0x95a5a6     # 灰色
        }
        return colors.get(step, 0xffffff)

async def test_discord():
    """测试Discord Bot功能"""
    logger.info("=== 开始Discord Bot测试 ===")
    
    # 测试网络连接
    if not test_discord_connection():
        logger.error("Discord连接测试失败，请检查网络和代理设置")
        return
        
    try:
        # 创建Bot实例
        bot = TestBot()
        
        # 设置重连策略
        bot.max_reconnect_delay = 30  # 最大重连延迟（秒）
        bot.reconnect_attempts = 5    # 最大重连尝试次数
        
        # 启动Bot
        try:
            logger.info("正在启动Bot...")
            await bot.start(BOT_TOKEN)
        except discord.LoginFailure as e:
            logger.error(f"Bot登录失败: {e}")
            return
        except discord.ConnectionClosed as e:
            logger.error(f"连接关闭: {e}")
            return
        except Exception as e:
            logger.error(f"启动过程中发生错误: {e}")
            return
            
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.error("详细错误信息:", exc_info=True)
    finally:
        # 确保资源被正确清理
        try:
            await bot.close()
        except:
            pass
        logger.info("=== Discord Bot测试结束 ===")

if __name__ == "__main__":
    # 设置事件循环策略
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行测试
    try:
        asyncio.run(test_discord())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.error("详细错误信息:", exc_info=True) 
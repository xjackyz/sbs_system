import discord
import asyncio
import logging
import aiohttp
import os
import sys

# 配置日志
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
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 从环境变量获取

# Webhook URLs
MONITOR_WEBHOOK = os.getenv("DISCORD_MONITOR_WEBHOOK")  # 从环境变量获取
SIGNAL_WEBHOOK = os.getenv("DISCORD_SIGNAL_WEBHOOK")   # 从环境变量获取
UPLOAD_WEBHOOK = os.getenv("DISCORD_UPLOAD_WEBHOOK")   # 从环境变量获取

# 代理配置
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"

# 设置系统代理
os.environ['ALL_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL

class SimpleBot(discord.Client):
    def __init__(self):
        logger.info("初始化 SimpleBot...")
        intents = discord.Intents.all()
        super().__init__(intents=intents)
            
    async def setup_hook(self) -> None:
        logger.info("开始设置Bot...")
        try:
            # 配置 connector
            connector = aiohttp.TCPConnector(
                ssl=False,
                verify_ssl=False,
                force_close=True,
                enable_cleanup_closed=True,
                use_dns_cache=False
            )
            
            # 创建session
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60),
                trust_env=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            
            # 替换 discord.py 的默认 session
            if hasattr(self.http, '_HTTPClient__session'):
                await self.http._HTTPClient__session.close()
                self.http._HTTPClient__session = self.session
            
            logger.info("Bot设置完成")
        except Exception as e:
            logger.error(f"Bot设置失败: {e}")
            logger.exception("详细错误信息:")
            raise
            
    async def on_connect(self):
        """当 Bot 连接到 Discord 时调用"""
        logger.info("Bot 已连接到 Discord!")
        
    async def on_disconnect(self):
        """当 Bot 断开连接时调用"""
        logger.warning("Bot 与 Discord 断开连接!")
            
    async def on_ready(self):
        try:
            logger.info(f"Bot {self.user} (ID: {self.user.id}) 已准备就绪!")
            logger.info(f"已连接到 {len(self.guilds)} 个服务器:")
            
            if not self.guilds:
                logger.warning("Bot 未加入任何服务器!")
                return
            
            for guild in self.guilds:
                logger.info(f"- 服务器: {guild.name} (ID: {guild.id})")
                logger.info(f"  - 成员数: {guild.member_count}")
                logger.info(f"  - 频道数: {len(guild.channels)}")
                
                # 在每个服务器中查找可以发送消息的文本频道
                text_channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
                
                if not text_channels:
                    logger.warning(f"在服务器 {guild.name} 中没有可用的文本频道!")
                    continue
                
                channel = text_channels[0]
                logger.info(f"选择频道 {channel.name} 发送消息")
                
                try:
                    # 发送测试消息
                    embed = discord.Embed(
                        title="机器人测试",
                        description="我已经成功连接到服务器！\n我可以：\n1. 接收和分析交易图表\n2. 生成交易信号\n3. 发送系统状态报告",
                        color=0x3498db
                    )
                    await channel.send(embed=embed)
                    logger.info(f"已在 {channel.name} 发送测试消息")
                    
                except discord.Forbidden as e:
                    logger.error(f"没有权限在 {channel.name} 发送消息: {e}")
                except discord.HTTPException as e:
                    logger.error(f"发送消息到 {channel.name} 失败: {e}")
                except Exception as e:
                    logger.error(f"在 {channel.name} 发送消息时发生错误: {e}")
                    logger.exception("详细错误信息:")
                
        except Exception as e:
            logger.error(f"on_ready 处理过程中发生错误: {e}")
            logger.exception("详细错误信息:")
            
    async def close(self):
        if hasattr(self, 'session'):
            await self.session.close()
        await super().close()

async def main():
    # 创建机器人实例
    client = SimpleBot()
    
    try:
        logger.info("正在启动Bot...")
        await client.start(TOKEN)
    except discord.LoginFailure as e:
        logger.error(f"登录失败，Token可能无效: {e}")
    except discord.HTTPException as e:
        logger.error(f"HTTP请求失败: {e}")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        logger.exception("详细错误信息:")
    finally:
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.exception("详细错误信息:") 
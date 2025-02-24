import discord
import logging
import os

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Discord配置
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 从环境变量获取

class MinimalBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
    async def on_ready(self):
        logger.info(f"Bot {self.user} 已准备就绪!")
        
    async def on_message(self, message):
        if message.author == self.user:
            return
            
        logger.info(f"收到消息: {message.content}")
        await message.channel.send("收到!")

client = MinimalBot()
client.run(TOKEN) 
import aiohttp
import asyncio
import logging
import json
import os

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Discord Token
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 从环境变量获取

# Intents
INTENTS = (1 << 0)  # GUILDS
INTENTS |= (1 << 1)  # GUILD_MEMBERS
INTENTS |= (1 << 9)  # GUILD_MESSAGES
INTENTS |= (1 << 15)  # MESSAGE_CONTENT

# 代理配置
PROXY_URL = "http://127.0.0.1:7897"
os.environ['ALL_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

async def test_gateway():
    try:
        # 配置 connector
        connector = aiohttp.TCPConnector(
            ssl=True,
            force_close=False,
            enable_cleanup_closed=True,
            verify_ssl=True,
            ttl_dns_cache=300,
            limit=100,
            keepalive_timeout=60
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=300,
                connect=60,
                sock_connect=60,
                sock_read=60
            ),
            trust_env=True
        ) as session:
            # 获取 Gateway URL
            logger.info("正在获取 Gateway URL...")
            async with session.get(
                'https://discord.com/api/v10/gateway',
                proxy=PROXY_URL,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"获取 Gateway URL 失败: {response.status}")
                    return
                    
                data = await response.json()
                gateway_url = data['url']
                logger.info(f"获取到 Gateway URL: {gateway_url}")
                
                # 连接到 WebSocket
                logger.info("正在连接到 WebSocket...")
                async with session.ws_connect(
                    gateway_url,
                    proxy=PROXY_URL,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                ) as ws:
                    logger.info("WebSocket 连接已建立")
                    
                    # 发送身份验证
                    auth_payload = {
                        "op": 2,
                        "d": {
                            "token": TOKEN,
                            "intents": INTENTS,
                            "properties": {
                                "os": "linux",
                                "browser": "discord.py",
                                "device": "discord.py"
                            }
                        }
                    }
                    
                    await ws.send_json(auth_payload)
                    logger.info("已发送身份验证消息")
                    
                    # 等待响应
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            logger.info(f"收到消息: {data}")
                            
                            # 如果收到心跳请求，发送心跳响应
                            if data.get('op') == 1:
                                await ws.send_json({
                                    "op": 1,
                                    "d": None
                                })
                                logger.info("已发送心跳响应")
                                
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket 错误: {msg.data}")
                            break
                            
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket 连接已关闭")
                            break
                            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.exception("详细错误信息:")

if __name__ == "__main__":
    asyncio.run(test_gateway()) 
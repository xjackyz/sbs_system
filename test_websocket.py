import asyncio
import aiohttp
import logging
import json
import os

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 代理配置
PROXY_URL = "http://127.0.0.1:7897"
os.environ['ALL_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL

async def test_discord_websocket():
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
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            # 设置代理
            session._proxy = PROXY_URL
            session._proxy_auth = None
            
            # 获取WebSocket网关URL
            logger.info("获取Discord网关URL...")
            async with session.get('https://discord.com/api/v10/gateway') as response:
                data = await response.json()
                gateway_url = data['url']
                logger.info(f"获取到网关URL: {gateway_url}")
                
            # 连接到WebSocket
            logger.info("正在连接到WebSocket...")
            async with session.ws_connect(
                gateway_url,
                proxy=PROXY_URL,
                proxy_headers=None
            ) as ws:
                logger.info("WebSocket连接已建立")
                
                # 发送身份验证消息
                auth_payload = {
                    "op": 2,
                    "d": {
                        "token": os.getenv("DISCORD_BOT_TOKEN"),
                        "intents": 513,
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
                        if data['op'] == 1:
                            await ws.send_json({
                                "op": 1,
                                "d": None
                            })
                            logger.info("已发送心跳响应")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket错误: {msg.data}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket连接已关闭")
                        break
                        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.exception("详细错误信息:")

if __name__ == "__main__":
    try:
        asyncio.run(test_discord_websocket())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.exception("详细错误信息:") 
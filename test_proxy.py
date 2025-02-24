import requests
import logging
import os
import json

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 代理配置
PROXY_URL = "http://127.0.0.1:7897"
proxies = {
    'http': PROXY_URL,
    'https': PROXY_URL
}

def test_proxy():
    try:
        # 测试 Discord API
        logger.info("测试 Discord API 连接...")
        response = requests.get(
            'https://discord.com/api/v10/gateway',
            proxies=proxies,
            timeout=10,
            verify=False
        )
        logger.info(f"Discord API 响应: {response.status_code} - {response.text}")

        # 测试 Discord Gateway
        logger.info("测试 Discord Gateway 连接...")
        response = requests.get(
            'https://gateway.discord.gg',
            proxies=proxies,
            timeout=10,
            verify=False
        )
        logger.info(f"Discord Gateway 响应: {response.status_code}")

        # 获取代理状态
        logger.info("获取 Clash 代理状态...")
        response = requests.get(
            'http://127.0.0.1:7897/proxies',
            timeout=5
        )
        logger.info(f"Clash 代理状态: {json.dumps(response.json(), indent=2)}")

    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")

if __name__ == "__main__":
    test_proxy() 
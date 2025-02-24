"""Discord功能测试"""
import asyncio
import pytest
from pathlib import Path
import os
import logging
from datetime import datetime

from src.utils.discord_handler import DiscordHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试配置
TEST_CONFIG = {
    'webhooks': {
        'monitor': "https://discord.com/api/webhooks/1342330366484283403/I2A7VgL_E6nHH6TvmOFDGYqLspW6Am35Vk1veP_RTywCkyMgCiru-wNhwjZoZF5wi13U",
        'signal': "https://discord.com/api/webhooks/1342330389963870240/CLj52_oZZ_ZJi2JvZECydc36nVdtjYF6l7uBG3NiMqi3pzOOWwQlmCQjsomjWxXtsY1R",
        'debug': "https://discord.com/api/webhooks/1342330356132614276/_-H_71ohRH1FH_xA3uEN-ttTDiuXtMxhwy91xx9Eta8RJpE1zZwVIgTqlIR0hEvlGEp8"
    },
    'bot_avatar': "https://cdn.discordapp.com/avatars/1343285001973792899/your_avatar_hash.png"  # 替换为实际的机器人头像URL
}

@pytest.fixture
def discord_handler():
    """创建Discord处理器实例"""
    return DiscordHandler(TEST_CONFIG)

@pytest.mark.asyncio
async def test_connection(discord_handler):
    """测试Discord连接"""
    result = await discord_handler.test_connection()
    assert result is True, "Discord连接测试失败"

@pytest.mark.asyncio
async def test_send_image_analysis(discord_handler):
    """测试发送图片分析结果"""
    # 准备测试数据
    test_image = "tests/data/test_chart.png"
    
    # 确保测试图片存在
    if not Path(test_image).exists():
        pytest.skip("测试图片不存在")
        
    analysis_result = {
        'sequence_type': 'SETUP',
        'confidence': 0.95,
        'timestamp': datetime.now().isoformat(),
        'details': '在当前价格位置发现潜在的建仓机会，市场结构显示上升趋势开始形成。'
    }
    
    result = await discord_handler.send_image_analysis(
        image_path=test_image,
        analysis_result=analysis_result
    )
    assert result is True, "发送图片分析结果失败"

@pytest.mark.asyncio
async def test_send_alert(discord_handler):
    """测试发送交易提醒"""
    alert_data = {
        'symbol': 'BTC/USDT',
        'action': 'BUY',
        'price': 45000.00,
        'stop_loss': 44000.00,
        'take_profit': 48000.00,
        'timestamp': datetime.now().isoformat()
    }
    
    result = await discord_handler.send_alert(alert_data)
    assert result is True, "发送交易提醒失败"

@pytest.mark.asyncio
async def test_multiple_messages(discord_handler):
    """测试发送多条消息"""
    # 准备测试数据
    alerts = [
        {
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'price': 45000.00
        },
        {
            'symbol': 'ETH/USDT',
            'action': 'SELL',
            'price': 2500.00
        }
    ]
    
    # 并发发送消息
    tasks = [
        discord_handler.send_alert(alert)
        for alert in alerts
    ]
    
    results = await asyncio.gather(*tasks)
    assert all(results), "发送多条消息失败"

@pytest.mark.asyncio
async def test_error_handling(discord_handler):
    """测试错误处理"""
    # 测试无效的webhook
    invalid_config = {
        'webhooks': {'signal': 'invalid_url'},
        'bot_avatar': None
    }
    invalid_handler = DiscordHandler(invalid_config)
    
    result = await invalid_handler.test_connection()
    assert result is False, "应该处理无效webhook的情况"
    
    # 测试无效的图片路径
    result = await discord_handler.send_image_analysis(
        image_path="invalid/path.png",
        analysis_result={}
    )
    assert result is False, "应该处理无效图片路径的情况"

if __name__ == '__main__':
    # 创建测试数据目录
    os.makedirs('tests/data', exist_ok=True)
    
    # 运行测试
    pytest.main(['-v', __file__]) 
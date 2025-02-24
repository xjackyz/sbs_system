#!/bin/bash

# 设置错误时退出
set -e

# 创建必要的目录
mkdir -p /app/logs /app/temp /app/models /app/data/historical_charts /app/screenshots

# 检查环境变量
if [ -z "$DISCORD_BOT_TOKEN" ]; then
    echo "错误: 未设置DISCORD_BOT_TOKEN"
    exit 1
fi

# 检查模型文件
if [ ! -d "/app/models/llava-sbs" ]; then
    echo "错误: 模型文件不存在"
    exit 1
fi

# 检查代理连接
echo "检查代理连接..."
if curl -s --connect-timeout 5 -x $http_proxy https://api.discord.com > /dev/null; then
    echo "代理连接正常"
else
    echo "警告: 代理连接可能存在问题"
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检查GPU状态..."
    nvidia-smi
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
fi

# 启动自监督学习服务（后台运行）
if [ "${ENABLE_SELF_SUPERVISED:-true}" = "true" ]; then
    echo "启动自监督学习服务..."
    python3 -m src.training.self_supervised &
fi

# 启动优化服务（后台运行）
if [ "${ENABLE_OPTIMIZATION:-true}" = "true" ]; then
    echo "启动模型优化服务..."
    python3 -m src.optimization.model_optimizer &
fi

# 启动测试监控（后台运行）
if [ "${ENABLE_TESTING:-true}" = "true" ]; then
    echo "启动测试监控服务..."
    python3 -m src.testing.monitor &
fi

# 启动主Bot服务
echo "启动Discord Bot..."
exec python3 -m src.bot.discord_bot 
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

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检查GPU状态..."
    nvidia-smi
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
fi

# 启动Discord Bot
echo "启动Discord Bot..."
cd /app
python3 src/main.py 
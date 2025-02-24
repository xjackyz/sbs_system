"""
健康检查服务
"""
import os
import psutil
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import start_http_server, Gauge
import uvicorn
import logging
from datetime import datetime

# 设置指标
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU内存使用量')
cpu_usage = Gauge('cpu_usage_percent', 'CPU使用率')
memory_usage = Gauge('memory_usage_bytes', '内存使用量')
uptime = Gauge('bot_uptime_seconds', '运行时间')

# 创建FastAPI应用
app = FastAPI()
start_time = datetime.now()

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_used.set(gpu_memory)
        
        # 检查CPU和内存
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        cpu_usage.set(cpu_percent)
        memory_usage.set(memory_info.rss)
        
        # 计算运行时间
        current_uptime = (datetime.now() - start_time).total_seconds()
        uptime.set(current_uptime)
        
        return {
            "status": "healthy",
            "gpu": {
                "available": torch.cuda.is_available(),
                "memory_used": gpu_memory if torch.cuda.is_available() else None
            },
            "cpu_usage": cpu_percent,
            "memory_usage": memory_info.rss,
            "uptime": current_uptime
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """主函数"""
    # 启动Prometheus指标服务器
    start_http_server(8000)
    
    # 启动FastAPI服务
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main() 
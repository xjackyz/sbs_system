from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import docker
import psutil
import json
from typing import Dict, List
import asyncio
from datetime import datetime

app = FastAPI(title="SBS System Web UI")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# Docker客户端
docker_client = docker.from_env()

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 获取Docker容器状态
        containers = docker_client.containers.list()
        container_status = []
        for container in containers:
            container_status.append({
                'name': container.name,
                'status': container.status,
                'image': container.image.tags[0] if container.image.tags else 'unknown'
            })
            
        return {
            'cpu': {
                'percent': cpu_percent
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'percent': disk.percent
            },
            'containers': container_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/stats")
async def get_analysis_stats():
    """获取分析统计信息"""
    try:
        # TODO: 从数据库或日志中获取实际统计数据
        return {
            'total_analyses': 100,
            'successful_signals': 75,
            'average_confidence': 0.85,
            'recent_signals': [
                {
                    'timestamp': '2024-02-24T10:00:00',
                    'symbol': 'NQ1!',
                    'type': '做多',
                    'confidence': 0.92
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs():
    """获取系统日志"""
    try:
        # TODO: 实现实际的日志获取逻辑
        return {
            'logs': [
                {
                    'timestamp': '2024-02-24T10:00:00',
                    'level': 'INFO',
                    'message': '系统启动成功'
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """实时日志WebSocket端点"""
    await websocket.accept()
    try:
        while True:
            # TODO: 实现实际的日志推送逻辑
            await websocket.send_json({
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': '系统运行正常'
            })
            await asyncio.sleep(5)
    except Exception as e:
        await websocket.close()

@app.get("/api/config")
async def get_config():
    """获取系统配置"""
    try:
        # TODO: 从配置文件中读取实际配置
        return {
            'model': {
                'base_model': 'models/llava-sbs',
                'device': 'cuda',
                'max_new_tokens': 1000
            },
            'system': {
                'log_level': 'INFO',
                'temp_dir': 'temp'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(config: Dict):
    """更新系统配置"""
    try:
        # TODO: 实现配置更新逻辑
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
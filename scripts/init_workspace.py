"""
初始化工作空间脚本
"""
import os
import shutil
from pathlib import Path

def init_workspace():
    """初始化工作空间"""
    # 创建必要的目录
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'logs',
        'models',
        'models/llava_sbs',
        'output',
        'temp',
        'cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 创建.env文件(如果不存在)
    if not os.path.exists('.env'):
        shutil.copy('.env.example', '.env')
        print("创建.env文件")
    
    # 检查模型目录
    model_dir = Path('models/llava_sbs')
    if not any(model_dir.iterdir()):
        print("警告: models/llava_sbs目录为空,请确保已下载模型文件")
    
    print("工作空间初始化完成")

if __name__ == '__main__':
    init_workspace() 
"""
初始化git仓库并上传到github
"""
import os
import subprocess
from pathlib import Path

def run_command(command):
    """运行shell命令"""
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"命令执行失败: {command}")
        print(f"错误信息: {process.stderr}")
        return False
    return True

def init_git():
    """初始化git仓库"""
    # 检查是否已经是git仓库
    if os.path.exists('.git'):
        print("已经是git仓库")
        return True
    
    # 初始化git仓库
    if not run_command('git init'):
        return False
    print("初始化git仓库成功")
    
    # 创建.gitignore文件
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 环境文件
.env
.venv
env/
venv/
ENV/

# 日志和数据
logs/
*.log
data/
temp/
cache/

# IDE
.idea/
.vscode/
*.swp
*.swo

# 模型文件
models/llava_sbs/

# 其他
.DS_Store
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("创建.gitignore文件成功")
    
    # 添加文件到git
    if not run_command('git add .'):
        return False
    print("添加文件到git成功")
    
    # 初始提交
    if not run_command('git commit -m "Initial commit"'):
        return False
    print("初始提交成功")
    
    return True

def setup_remote(repo_url):
    """设置远程仓库"""
    # 添加远程仓库
    if not run_command(f'git remote add origin {repo_url}'):
        return False
    print("添加远程仓库成功")
    
    # 推送到远程仓库
    if not run_command('git push -u origin master'):
        return False
    print("推送到远程仓库成功")
    
    return True

def main():
    """主函数"""
    # 初始化git仓库
    if not init_git():
        print("初始化git仓库失败")
        return
    
    # 获取github仓库URL
    repo_url = input("请输入github仓库URL: ")
    if not repo_url:
        print("未提供github仓库URL")
        return
    
    # 设置远程仓库
    if not setup_remote(repo_url):
        print("设置远程仓库失败")
        return
    
    print("初始化git仓库并上传到github成功")

if __name__ == '__main__':
    main() 
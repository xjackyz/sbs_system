FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY setup.py .

# 安装Python依赖
RUN pip install -r requirements.txt
RUN pip install -e .

# 复制源代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV ENVIRONMENT=development

# 启动命令
CMD ["python", "main.py"] 
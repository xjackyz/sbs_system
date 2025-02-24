# SBS系统部署指南

## 目录

1. [系统要求](#系统要求)
2. [安装步骤](#安装步骤)
3. [配置说明](#配置说明)
4. [运行说明](#运行说明)
5. [监控和维护](#监控和维护)
6. [故障排除](#故障排除)
7. [更新和升级](#更新和升级)

## 系统要求

### 硬件要求
- CPU: 8核或以上
- 内存: 32GB或以上
- 磁盘空间: 100GB或以上
- GPU: NVIDIA GPU (16GB显存或以上)

### 软件要求
- 操作系统: Ubuntu 22.04 LTS
- Python 3.8+
- CUDA 11.8
- Docker 24.0+
- NVIDIA Container Toolkit

## 安装步骤

### 1. 基础环境配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y build-essential git python3-pip

# 安装CUDA和NVIDIA驱动
# 请参考NVIDIA官方文档安装适合您系统的版本
```

### 2. 安装Docker

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. 克隆项目

```bash
git clone https://github.com/your-org/sbs_system.git
cd sbs_system
```

### 4. 配置环境变量

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑配置文件
nano .env

# 设置必要的环境变量
DISCORD_WEBHOOK_URL=your_webhook_url
LLAVA_MODEL_PATH=/path/to/model
```

### 5. 构建Docker镜像

```bash
# 开发环境
docker build -t sbs_system:dev -f docker/dev/Dockerfile .

# 生产环境
docker build -t sbs_system:prod -f docker/prod/Dockerfile .
```

## 配置说明

### 1. 环境变量配置

主要配置项说明：
- `DISCORD_WEBHOOK_URL`: Discord通知webhook地址
- `LLAVA_MODEL_PATH`: LLaVA模型路径
- `LOG_LEVEL`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `NUM_WORKERS`: 工作进程数
- `USE_GPU`: 是否使用GPU (True/False)

### 2. 系统配置

配置文件位置: `config/config.py`

主要配置项：
- 运行模式配置
- 预处理配置
- 模型配置
- 监控配置
- 验证配置

## 运行说明

### 1. 使用Docker运行

```bash
# 开发环境
docker run -d --gpus all \
    --name sbs_system_dev \
    -v $(pwd):/app \
    -v /path/to/model:/model \
    --env-file .env \
    sbs_system:dev

# 生产环境
docker run -d --gpus all \
    --name sbs_system_prod \
    -v /path/to/model:/model \
    --env-file .env \
    sbs_system:prod
```

### 2. 直接运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行系统
python main.py
```

## 监控和维护

### 1. 系统监控

- 使用系统内置的监控工具：
```bash
python scripts/check_system.py
```

- 查看监控日志：
```bash
tail -f logs/monitor_*.log
```

### 2. 日志管理

- 日志位置: `logs/`
- 日志轮转配置: `logging.conf`
- 日志分析工具: `scripts/analyze_logs.py`

### 3. 性能监控

- CPU和内存使用率监控
- GPU使用率监控
- 系统响应时间监控
- 数据处理速度监控

### 4. 数据备份

```bash
# 备份配置和数据
python scripts/backup.py

# 恢复数据
python scripts/restore.py --backup-file backup_20240101.tar.gz
```

## 故障排除

### 1. 常见问题

1. GPU内存不足
   - 检查显存使用情况
   - 调整批处理大小
   - 清理GPU缓存

2. 系统响应慢
   - 检查系统负载
   - 检查网络连接
   - 检查数据处理队列

3. 数据验证失败
   - 检查数据格式
   - 检查数据完整性
   - 查看验证日志

### 2. 错误处理

- 查看错误日志：
```bash
tail -f logs/error.log
```

- 运行诊断工具：
```bash
python scripts/diagnose.py
```

### 3. 系统恢复

```bash
# 停止服务
docker stop sbs_system_prod

# 清理缓存
python scripts/clean_cache.py

# 重启服务
docker start sbs_system_prod
```

## 更新和升级

### 1. 系统更新

```bash
# 拉取最新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade

# 重新构建Docker镜像
docker build -t sbs_system:prod -f docker/prod/Dockerfile .
```

### 2. 模型更新

```bash
# 下载新模型
python scripts/download_model.py --version latest

# 更新模型配置
python scripts/update_model_config.py
```

### 3. 配置更新

```bash
# 更新配置文件
python scripts/update_config.py

# 验证配置
python scripts/validate_config.py
``` 
FROM ubuntu:22.04

# 添加构建参数
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

# 设置阿里云镜像源并安装基础依赖
RUN set -ex \
    && unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY \
    && echo "Acquire::http::Proxy \"DIRECT\";" > /etc/apt/apt.conf.d/proxy.conf \
    && echo "Acquire::https::Proxy \"DIRECT\";" >> /etc/apt/apt.conf.d/proxy.conf \
    && echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        git \
        curl \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential \
        pkg-config \
        libcairo2-dev \
        libgirepository1.0-dev \
        libssl-dev \
        libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "基础依赖安装完成"

# 设置Python
RUN set -ex \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && echo "Python设置完成"

# 设置pip镜像源
RUN set -ex \
    && mkdir -p ~/.pip \
    && { \
        echo '[global]'; \
        echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple'; \
        echo 'trusted-host = pypi.tuna.tsinghua.edu.cn'; \
        echo 'timeout = 120'; \
        echo 'retries = 3'; \
        echo 'proxy = '; \
    } > ~/.pip/pip.conf \
    && echo "pip配置完成"

# 升级pip并安装基本包
RUN set -ex \
    && unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && echo "pip升级完成"

# 安装Python依赖
COPY requirements.txt .
RUN set -ex \
    && unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY \
    && pip install --no-cache-dir -r requirements.txt \
    && echo "Python依赖安装完成"

# 复制项目文件
COPY . .

# 创建必要的目录
RUN set -ex \
    && mkdir -p logs temp models data/historical_charts screenshots monitoring \
    && echo "目录创建完成"

# 设置脚本权限
RUN set -ex \
    && chmod +x scripts/start.sh \
    && echo "脚本权限设置完成"

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["./scripts/start.sh"] 
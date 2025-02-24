# 配置文档

## 系统配置

系统配置文件位于 `config/system_config.yaml`。

### 基本配置

```yaml
# 系统配置
system:
  # 运行设备
  device: 'cuda'
  # 日志目录
  log_dir: 'logs'
  # 缓存目录
  cache_dir: 'cache'
  # 调试模式
  debug: false

# 模型配置
model:
  # 基础模型
  base_model: 'llava-v1.6'
  # 预训练权重路径
  weights_path: 'models/llava-sbs'
  # 是否使用量化
  use_quantization: true
  # 是否使用剪枝
  use_pruning: true
  # 是否使用知识蒸馏
  use_distillation: false
  # 视觉塔
  vision_tower: 'openai/clip-vit-large-patch14-336'
  # 图像大小
  image_size: [336, 336]
  # 最大长度
  max_length: 4096
  # 数据类型
  dtype: 'float16'
```

### 数据配置

```yaml
# 数据配置
data:
  # 图像大小
  image_size: [336, 336]
  # 数据目录
  data_dir: 'data'
  # 历史数据目录
  historical_data_dir: 'data/historical'
  # 实时数据目录
  realtime_data_dir: 'data/realtime'
  # 缓存设置
  cache:
    enabled: true
    max_size: 1000
    ttl: 3600
```

### 交易提醒配置

```yaml
# 交易提醒配置
alerts:
  # 配置文件路径
  config_path: 'config/alerts_config.json'
  # 置信度阈值
  confidence_threshold: 0.8
  # 保存目录
  save_dir: 'alerts'
  # 通知设置
  notifications:
    # 钉钉
    dingtalk:
      enabled: false
      webhook: ''
      secret: ''
    # 企业微信
    wecom:
      enabled: false
      webhook: ''
    # Telegram
    telegram:
      enabled: false
      bot_token: ''
      chat_id: ''
```

## 训练配置

训练配置文件位于 `config/training_config.yaml`。

### 模型配置

```yaml
# 模型配置
model:
  # 基础模型选择
  base_model: 'resnet50'
  # 是否使用预训练权重
  pretrained: true
  # 是否冻结主干网络
  freeze_backbone: true
```

### 数据配置

```yaml
# 数据配置
data:
  # 图像大小
  image_size: [224, 224]
  # 训练数据目录
  train_dir: 'data/train'
  # 验证数据目录
  val_dir: 'data/val'
  # 测试数据目录
  test_dir: 'data/test'
  # 数据增强
  augmentation:
    horizontal_flip: true
    vertical_flip: false
    random_rotation: 10
    random_brightness: 0.1
    random_contrast: 0.1
```

### 训练配置

```yaml
# 训练配置
training:
  # 设备
  device: 'cuda'
  # 随机种子
  seed: 42
  # 批次大小
  batch_size: 32
  # 训练轮数
  num_epochs: 50
  # 学习率
  learning_rate: 0.001
  # 权重衰减
  weight_decay: 0.0001
  # 数据加载器线程数
  num_workers: 4
```

## 优化配置

优化配置文件位于 `config/optimization_config.yaml`。

### 量化配置

```yaml
# 量化配置
quantization:
  # 量化方法: 'dynamic' 或 'static'
  method: 'dynamic'
  # 静态量化配置
  static:
    # 量化配置后端
    backend: 'fbgemm'
    # 是否保留float32的权重
    preserve_dtype: false
    # 量化精度
    dtype: 'qint8'
```

### 剪枝配置

```yaml
# 剪枝配置
pruning:
  # 剪枝方法: 'unstructured' 或 'structured'
  method: 'unstructured'
  # 剪枝比例 (0-1)
  amount: 0.3
  # 需要剪枝的层
  target_layers:
    - 'conv1'
    - 'conv2'
    - 'fc1'
    - 'fc2'
```

## 环境变量

环境变量可以通过 `.env` 文件配置，示例见 `.env.example`：

```bash
# 系统配置
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# 设备配置
USE_GPU=true
NUM_WORKERS=4

# 模型配置
MODEL_PATH=models/llava-sbs
VISION_MODEL_PATH=openai/clip-vit-large-patch14-336
```

## 配置最佳实践

1. 敏感信息（如API密钥）应该通过环境变量配置
2. 不同环境使用不同的配置文件
3. 配置文件应该版本控制，但不包含敏感信息
4. 使用配置验证确保配置正确性
5. 为配置提供合理的默认值 
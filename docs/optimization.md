# 模型优化文档

本文档介绍了 SBS Trading System 中使用的各种模型优化技术。

## 量化优化

### 动态量化

动态量化是最简单的量化方法，它在运行时将权重从 FP32 转换为 INT8。

```python
from src.optimization.quantization import ModelQuantizer

# 初始化量化器
quantizer = ModelQuantizer(model)

# 应用动态量化
quantized_model = quantizer.dynamic_quantize()
```

### 静态量化

静态量化需要校准数据来确定最佳的量化参数。

```python
# 准备校准数据
calibration_data = get_calibration_data()

# 配置静态量化
config = {
    'backend': 'fbgemm',
    'dtype': 'qint8'
}

# 应用静态量化
quantized_model = quantizer.static_quantize(
    calibration_data=calibration_data,
    config=config
)
```

## 模型剪枝

### 非结构化剪枝

非结构化剪枝可以移除不重要的权重。

```python
from src.optimization.pruning import ModelPruner

# 初始化剪枝器
pruner = ModelPruner(model)

# 应用非结构化剪枝
results = pruner.unstructured_pruning(
    layer_names=['conv1', 'conv2', 'fc1'],
    amount=0.3
)
```

### 结构化剪枝

结构化剪枝可以移除整个通道或神经元。

```python
# 应用结构化剪枝
results = pruner.structured_pruning(
    layer_names=['conv1', 'conv2'],
    amount=0.3,
    dim=0  # 按输出通道剪枝
)
```

## 知识蒸馏

使用教师模型指导学生模型的训练。

```python
from src.optimization.distillation import KnowledgeDistillation

# 初始化知识蒸馏
distillation = KnowledgeDistillation(
    teacher_model=teacher,
    student_model=student,
    optimizer=optimizer
)

# 训练学生模型
for batch_data, labels in train_loader:
    results = distillation.train_step(batch_data, labels)
```

## 性能评估

### 模型大小

```python
# 评估量化效果
performance = quantizer.evaluate_performance(
    test_data=test_data,
    original_model=model
)

print(f"原始模型大小: {performance['original_size_mb']:.2f} MB")
print(f"量化后大小: {performance['quantized_size_mb']:.2f} MB")
print(f"压缩比: {performance['compression_ratio']:.2f}x")
```

### 剪枝统计

```python
# 获取剪枝统计信息
stats = pruner.get_pruning_statistics()

print(f"全局稀疏度: {stats['global_sparsity']:.2f}%")
print(f"零参数数量: {stats['zero_parameters']}")
print(f"总参数数量: {stats['total_parameters']}")
```

## 优化建议

1. **量化选择**
   - 对于推理任务，优先使用静态量化
   - 对于训练任务，可以考虑动态量化
   - 注意量化对精度的影响

2. **剪枝策略**
   - 从小比例开始尝试（如 10-30%）
   - 优先剪枝冗余层
   - 监控精度变化
   - 考虑使用迭代式剪枝

3. **知识蒸馏**
   - 选择合适的教师模型
   - 调整温度参数
   - 平衡硬目标和软目标的损失权重
   - 使用适当的数据增强

4. **综合优化**
   - 可以组合使用多种优化方法
   - 先剪枝后量化
   - 使用知识蒸馏提升性能
   - 持续监控和评估

## 性能监控

### 资源使用

```python
# 监控 GPU 内存
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

# 监控 CPU 内存
import psutil
psutil.Process().memory_info().rss
```

### 推理速度

```python
import time

start_time = time.time()
output = model(input_data)
inference_time = time.time() - start_time
```

## 故障排除

1. **量化问题**
   - 检查量化配置
   - 验证校准数据
   - 监控数值范围
   - 检查算子支持

2. **剪枝问题**
   - 验证剪枝比例
   - 检查层名称
   - 监控精度变化
   - 确保模型可训练

3. **知识蒸馏问题**
   - 检查教师模型
   - 调整温度参数
   - 验证损失计算
   - 监控学习过程

## 最佳实践

1. **渐进式优化**
   - 从单一方法开始
   - 逐步增加优化强度
   - 持续监控性能
   - 保存中间结果

2. **性能平衡**
   - 权衡速度和精度
   - 考虑资源限制
   - 适应具体场景
   - 保持可维护性

3. **验证流程**
   - 建立基准测试
   - 进行对比实验
   - 记录优化过程
   - 文档化结果 
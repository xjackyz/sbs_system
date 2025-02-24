# API 文档

## 模型 API

### SBSAnalyzer

主要的分析模型类，用于处理和分析交易图表。

```python
class SBSAnalyzer:
    def __init__(self,
                 base_model: str = 'resnet50',
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 num_classes: int = 5,
                 device: str = 'cuda'):
        """初始化 SBS 分析器

        Args:
            base_model: 基础模型名称 ('resnet50', 'vgg16' 等)
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结主干网络
            num_classes: 分类数量（SBS五个步骤）
            device: 运行设备
        """
        pass

    def predict_step(self,
                    image: torch.Tensor) -> Tuple[SBSSteps, float]:
        """预测图像的 SBS 步骤

        Args:
            image: 输入图像张量

        Returns:
            预测的 SBS 步骤和置信度
        """
        pass
```

### LLaVAAnalyzer

LLaVA 模型分析器，用于高级图表分析。

```python
class LLaVAAnalyzer:
    def __init__(self, config: Config):
        """初始化 LLaVA 分析器

        Args:
            config: 配置对象
        """
        pass

    def analyze_image(self,
                     image_path: str,
                     prompt: str) -> str:
        """分析图像

        Args:
            image_path: 图像路径
            prompt: 提示文本

        Returns:
            分析结果
        """
        pass
```

## 优化 API

### ModelQuantizer

模型量化工具。

```python
class ModelQuantizer:
    def __init__(self, model: torch.nn.Module):
        """初始化量化器

        Args:
            model: 要量化的模型
        """
        pass

    def dynamic_quantize(self) -> torch.nn.Module:
        """动态量化模型"""
        pass

    def static_quantize(self,
                       calibration_data: torch.Tensor,
                       config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """静态量化模型

        Args:
            calibration_data: 校准数据
            config: 量化配置
        """
        pass
```

### ModelPruner

模型剪枝工具。

```python
class ModelPruner:
    def __init__(self, model: torch.nn.Module):
        """初始化剪枝器

        Args:
            model: 要剪枝的模型
        """
        pass

    def unstructured_pruning(self,
                            layer_names: List[str],
                            amount: float = 0.3) -> Dict[str, float]:
        """非结构化剪枝

        Args:
            layer_names: 需要剪枝的层名称列表
            amount: 剪枝比例（0-1之间）
        """
        pass
```

## 工具 API

### TradingAlertManager

交易提醒管理器。

```python
class TradingAlertManager:
    def __init__(self, config_path: Optional[str] = None):
        """初始化交易提醒管理器

        Args:
            config_path: 配置文件路径
        """
        pass

    def generate_alert(self,
                      step: SBSSteps,
                      confidence: float,
                      symbol: str,
                      timeframe: str) -> Dict[str, Any]:
        """生成交易提醒

        Args:
            step: 预测的SBS步骤
            confidence: 预测置信度
            symbol: 交易对
            timeframe: 时间周期
        """
        pass
```

## 批处理 API

### BatchAnalyzer

批量分析工具。

```python
class BatchAnalyzer:
    def __init__(self,
                 model_path: str,
                 device: str = "cuda"):
        """初始化批量分析器

        Args:
            model_path: 模型路径
            device: 运行设备
        """
        pass

    def analyze_directory(self,
                         input_dir: str) -> List[Dict[str, Any]]:
        """分析目录中的所有图片

        Args:
            input_dir: 输入目录路径

        Returns:
            分析结果列表
        """
        pass
```

## 错误处理

所有 API 在遇到错误时会抛出以下异常：

- `ModelError`: 模型相关错误
- `ConfigError`: 配置相关错误
- `DataError`: 数据相关错误
- `SystemError`: 系统相关错误

## 使用示例

```python
from src.model.sbs_analyzer import SBSAnalyzer
from src.utils.trading_alerts import TradingAlertManager

# 初始化模型
model = SBSAnalyzer(base_model='resnet50', pretrained=True)

# 分析图像
step, confidence = model.predict_step(image)

# 生成提醒
alert_manager = TradingAlertManager()
alert = alert_manager.generate_alert(
    step=step,
    confidence=confidence,
    symbol="BTC/USDT",
    timeframe="1h"
) 
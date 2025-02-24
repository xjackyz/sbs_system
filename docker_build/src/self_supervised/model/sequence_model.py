import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置"""
    image_size: Tuple[int, int] = (224, 224)  # 输入图像尺寸
    sequence_length: int = 100                 # 序列长度
    hidden_size: int = 256                    # 隐藏层大小
    num_heads: int = 8                        # 注意力头数
    num_layers: int = 6                       # Transformer层数
    dropout: float = 0.1                      # Dropout比率
    num_classes: int = 3                      # 序列类型数量
    num_points: int = 5                       # 关键点数量

class MultiScaleAttention(nn.Module):
    """多尺度注意力模块"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 多尺度特征提取
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # 不同尺度的卷积核
        ])
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, L, H]
            
        Returns:
            融合后的特征 [B, L, H]
        """
        B, L, H = x.shape
        
        # 多尺度特征提取
        multi_scale_features = []
        x_conv = x.transpose(1, 2)  # [B, H, L]
        
        for conv in self.scale_convs:
            scale_feat = conv(x_conv)  # [B, H, L]
            multi_scale_features.append(scale_feat.transpose(1, 2))  # [B, L, H]
        
        # 自注意力计算
        x = x.transpose(0, 1)  # [L, B, H]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)  # [B, L, H]
        
        # 特征融合
        multi_scale_features.append(attn_output)
        fused_features = torch.cat(multi_scale_features, dim=-1)  # [B, L, H*3]
        output = self.fusion(fused_features)  # [B, L, H]
        
        return output

class TemporalAttention(nn.Module):
    """时序注意力模块"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 位置编码
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, 100, hidden_size)  # 最大序列长度100
        )
        
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, L, H]
            mask: 注意力掩码 [B, L]
            
        Returns:
            时序注意力特征 [B, L, H]
        """
        B, L, H = x.shape
        
        # 添加位置编码
        x = x + self.pos_encoder[:, :L, :]
        
        # 计算时序注意力
        x = x.transpose(0, 1)  # [L, B, H]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, L]
            
        temporal_output, _ = self.temporal_attention(x, x, x, key_padding_mask=mask)
        temporal_output = temporal_output.transpose(0, 1)  # [B, L, H]
        
        # 输出投影
        output = self.output_projection(temporal_output)
        
        return output

class SequenceModel(nn.Module):
    """SBS序列模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # 图像特征提取器 (使用预训练的ResNet)
        self.image_encoder = self._create_image_encoder()
        
        # 序列特征提取器
        self.sequence_encoder = self._create_sequence_encoder()
        
        # 多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(
            self.config.hidden_size,
            self.config.num_heads,
            self.config.dropout
        )
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(
            self.config.hidden_size,
            self.config.num_heads,
            self.config.dropout
        )
        
        # 多模态融合Transformer
        self.fusion_transformer = self._create_fusion_transformer()
        
        # 序列分类头
        self.sequence_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.num_classes)
        )
        
        # 关键点定位头
        self.point_locator = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.num_points * 2)  # x,y坐标
        )
        
        # 交易信号生成头
        self.signal_generator = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, 3)  # 买入/卖出/不操作
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info("序列模型初始化完成")
        
    def _create_image_encoder(self) -> nn.Module:
        """创建图像编码器"""
        # 使用预训练的ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # 冻结部分层
        for param in resnet.parameters():
            param.requires_grad = False
            
        # 修改最后的全连接层
        resnet.fc = nn.Linear(resnet.fc.in_features, self.config.hidden_size)
        
        return resnet
        
    def _create_sequence_encoder(self) -> nn.Module:
        """创建序列编码器"""
        return nn.Sequential(
            nn.Linear(5, self.config.hidden_size // 2),  # OHLCV数据
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_size // 2,
                    nhead=self.config.num_heads // 2,
                    dim_feedforward=self.config.hidden_size * 2,
                    dropout=self.config.dropout
                ),
                num_layers=self.config.num_layers // 2
            ),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size)
        )
        
    def _create_fusion_transformer(self) -> nn.Module:
        """创建多模态融合Transformer"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.hidden_size * 4,
                dropout=self.config.dropout
            ),
            num_layers=self.config.num_layers
        )
        
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, inputs: Dict) -> Dict:
        """
        前向传播
        
        Args:
            inputs: 输入数据字典
                - chart_image: 图表图像
                - sequence_data: 序列数据
                - sequence_type: 序列类型
                - sequence_id: 序列ID
                
        Returns:
            输出字典
                - sequence_logits: 序列分类logits
                - point_coords: 关键点坐标
                - trade_signals: 交易信号
        """
        # 1. 特征提取
        image_features = self.image_encoder(inputs['chart_image'])  # [B, H]
        sequence_features = self.sequence_encoder(inputs['sequence_data'])  # [B, L, H]
        
        # 2. 多尺度特征提取和注意力
        multi_scale_features = self.multi_scale_attention(sequence_features)
        
        # 3. 时序注意力
        temporal_features = self.temporal_attention(multi_scale_features)
        
        # 4. 特征融合
        B, L, H = temporal_features.shape
        image_features = image_features.unsqueeze(1).expand(-1, L, -1)
        fused_features = torch.cat([image_features, temporal_features], dim=-1)
        fused_features = self.fusion_transformer(fused_features)
        
        # 5. 多任务输出
        # 序列分类
        sequence_features = torch.mean(fused_features, dim=1)  # [B, H]
        sequence_logits = self.sequence_classifier(sequence_features)
        
        # 关键点定位
        point_features = torch.max(fused_features, dim=1)[0]  # [B, H]
        point_coords = self.point_locator(point_features)
        point_coords = point_coords.view(-1, self.config.num_points, 2)
        
        # 交易信号生成
        signal_features = torch.cat([sequence_features, point_features], dim=-1)
        trade_signals = self.signal_generator(signal_features)
        
        return {
            'sequence_logits': sequence_logits,
            'point_coords': point_coords,
            'trade_signals': trade_signals
        }
        
    def train_stage1(self, mode: bool = True):
        """设置序列识别预训练阶段"""
        self.train(mode)
        # 冻结不相关的模块
        for param in self.point_locator.parameters():
            param.requires_grad = False
        for param in self.signal_generator.parameters():
            param.requires_grad = False
            
    def train_stage2(self, mode: bool = True):
        """设置关键点识别训练阶段"""
        self.train(mode)
        # 解冻关键点定位模块
        for param in self.point_locator.parameters():
            param.requires_grad = True
        # 冻结其他模块
        for param in self.sequence_classifier.parameters():
            param.requires_grad = False
        for param in self.signal_generator.parameters():
            param.requires_grad = False
            
    def train_stage3(self, mode: bool = True):
        """设置交易信号生成阶段"""
        self.train(mode)
        # 解冻所有模块
        for param in self.parameters():
            param.requires_grad = True 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

class DistillationLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 0.5, 
                 temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        """
        # 计算蒸馏损失（软目标）
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 计算学生模型的硬目标损失
        student_loss = F.cross_entropy(student_logits, labels)
        
        # 组合损失
        total_loss = (self.alpha * student_loss + 
                     (1 - self.alpha) * distillation_loss)
        
        return total_loss

class KnowledgeDistillation:
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Optional[nn.Module] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or DistillationLoss()
        self.device = device
        self.teacher_model.eval()  # 教师模型设置为评估模式
        
    def train_step(self, 
                  batch_data: torch.Tensor, 
                  labels: torch.Tensor) -> Dict[str, float]:
        """单步训练
        
        Args:
            batch_data: 输入数据批次
            labels: 真实标签
        """
        batch_data = batch_data.to(self.device)
        labels = labels.to(self.device)
        
        # 获取教师模型输出
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch_data)
            
        # 获取学生模型输出
        student_logits = self.student_model(batch_data)
        
        # 计算损失
        loss = self.loss_fn(student_logits, teacher_logits, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 计算准确率
        student_preds = torch.argmax(student_logits, dim=1)
        accuracy = (student_preds == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate(self, 
                val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估学生模型性能"""
        self.student_model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, labels in val_loader:
                batch_data = batch_data.to(self.device)
                labels = labels.to(self.device)
                
                teacher_logits = self.teacher_model(batch_data)
                student_logits = self.student_model(batch_data)
                
                loss = self.loss_fn(student_logits, teacher_logits, labels)
                total_loss += loss.item()
                
                student_preds = torch.argmax(student_logits, dim=1)
                accuracy = (student_preds == labels).float().mean().item()
                total_accuracy += accuracy
                
                num_batches += 1
        
        self.student_model.train()
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def save_student_model(self, path: str):
        """保存学生模型"""
        torch.save(self.student_model.state_dict(), path)
        
    def load_student_model(self, path: str):
        """加载学生模型"""
        self.student_model.load_state_dict(torch.load(path))
        
    def get_model_sizes(self) -> Tuple[float, float]:
        """获取教师和学生模型大小"""
        def get_model_size(model: nn.Module) -> float:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
            
        teacher_size = get_model_size(self.teacher_model)
        student_size = get_model_size(self.student_model)
        return teacher_size, student_size 
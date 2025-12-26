import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        """
        Focal Loss 支持 label_smoothing

        参数：
            alpha (Tensor): 类别权重，形状 [num_classes]
            gamma (float): 聚焦参数，越大越关注难样本
            label_smoothing (float): 标签平滑系数，取值范围 [0, 1]
            reduction: 'none'/'mean'/'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # 如果传入 alpha 则使用，否则生成一个全1的权重向量
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = None

        # 使用 CrossEntropyLoss 自动处理标签平滑
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=self.alpha, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )

    def forward(self, inputs, targets):
        """
        inputs: [B, C] 或 [B, C, H, W]，logits（未经过 softmax）
        targets: [B] 或 [B, H, W]，真实类别标签（long 类型）
        """
        num_classes = inputs.size(1)

        # 计算 CrossEntropyLoss（包含标签平滑）
        ce_loss = self.cross_entropy_loss(inputs, targets)

        # 计算 Focal Loss 中的权重
        prob = torch.exp(-ce_loss)  # 从NLL得到概率
        focal_weight = (1.0 - prob) ** self.gamma

        # 计算最终的 Focal Loss
        loss = focal_weight * ce_loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

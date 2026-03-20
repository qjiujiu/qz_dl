import torch
import torch.nn as nn
from src.utils.weight_perbutation.base import BaseAttacker
from typing import Optional


class FGM(BaseAttacker):
    def __init__(self, model: nn.Module, epsilon: float = 1.0, emb_name: str = 'embedding'):
        super().__init__(model, emb_name)
        self.epsilon = epsilon

    def attack(self, epsilon: Optional[float] = None, emb_name: Optional[str] = None):
        """
        :param epsilon: 如果为 None，则使用初始化时的 self.epsilon
        :param emb_name: 如果为 None，则使用初始化时的 self.emb_name
        """
        # 1. 确定当前使用的参数
        current_epsilon = epsilon if epsilon is not None else self.epsilon
        current_emb_name = emb_name if emb_name is not None else self.emb_name
        
        # 2. 寻找并备份 Embedding 参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and current_emb_name in name:
                # 只有还没备份过才备份 (防止重复 attack 导致覆盖原始备份)
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                
                # 3. 计算扰动
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = current_epsilon * param.grad / norm
                    param.data.add_(r_at)
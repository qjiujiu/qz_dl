import torch
import torch.nn as nn
from typing import Optional

class BaseAttacker:
    def __init__(self, model: nn.Module, emb_name: str = 'embedding'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, **kwargs):
        raise NotImplementedError

    def restore(self, emb_name: Optional[str] = None):
        # 允许 restore 时临时指定名字，否则用实例默认的
        target_name = emb_name if emb_name is not None else self.emb_name
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and target_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
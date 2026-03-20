import torch
import torch.nn as nn
from src.utils.weight_perbutation.base import BaseAttacker
from typing import Optional

class PGD(BaseAttacker):
    def __init__(self, model: nn.Module, epsilon: float = 0.01, alpha: float = 0.001, emb_name: str = 'embedding'):
        super().__init__(model, emb_name)
        self.epsilon = epsilon
        self.alpha = alpha

    def attack(self, epsilon: Optional[float] = None, alpha: Optional[float] = None, 
               emb_name: Optional[str] = None, is_first_attack: bool = False):
        
        # 确定当前使用的参数
        curr_eps = epsilon if epsilon is not None else self.epsilon
        curr_alpha = alpha if alpha is not None else self.alpha
        curr_emb_name = emb_name if emb_name is not None else self.emb_name

        for name, param in self.model.named_parameters():
            if param.requires_grad and curr_emb_name in name:
                
                # 备份逻辑：如果是第一次攻击 OR 还没备份过
                if is_first_attack or name not in self.backup:
                    self.backup[name] = param.data.clone()

                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 步长 alpha
                    r_at = curr_alpha * param.grad / norm
                    param.data.add_(r_at)
                    
                    # 投影 (Project)
                    param_data = param.data
                    orig_data = self.backup[name]
                    perturbation = param_data - orig_data
                    
                    if torch.norm(perturbation) > curr_eps:
                        r_at = curr_eps * perturbation / torch.norm(perturbation)
                        param.data = orig_data + r_at

# class PGD:
#     def __init__(self, model):
#         self.model = model
#         self.emb_backup = {}
#         self.grad_backup = {}

#     def attack(self, epsilon=0.01, alpha=0.001, emb_name='embedding', is_first_attack=False):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
                
#                 # 【核心修复】如果标记为第一次攻击，或者 发现没有备份过，都执行备份
#                 if is_first_attack or name not in self.emb_backup:
#                     self.emb_backup[name] = param.data.clone()

#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     # 步长 alpha
#                     r_at = alpha * param.grad / norm
#                     param.data.add_(r_at)
                    
#                     # 投影 (Project)
#                     param_data = param.data
#                     orig_data = self.emb_backup[name] # 现在这里肯定有值了
#                     perturbation = param_data - orig_data
                    
#                     if torch.norm(perturbation) > epsilon:
#                         r_at = epsilon * perturbation / torch.norm(perturbation)
#                         param.data = orig_data + r_at
#         for name, param in self.model.named_parameters():

#     def restore(self, emb_name='embedding'):
#             if param.requires_grad and emb_name in name:
#                 if name in self.emb_backup:
#                     param.data = self.emb_backup[name]
#         self.emb_backup = {}
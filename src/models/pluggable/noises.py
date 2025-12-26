from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import register_pluggable_module


import torch
import torch.nn as nn



@register_pluggable_module(PluginType.GaussLinf)
class GaussNoiseLinf(nn.Module):
    def __init__(self, epsilon=0.015, **kwargs):
        """ 添加一个满足约束的高斯噪声
            embed_dim: The dimensionality of the input embedding.
            epsilon: The maximum perturbation, i.e., the Linf norm constraint.
        """
        super(GaussNoiseLinf, self).__init__()
        self.epsilon = epsilon  # Maximum noise magnitude

    def forward(self, x):
        noise = torch.randn_like(x) * self.epsilon  
        noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        return x + noise
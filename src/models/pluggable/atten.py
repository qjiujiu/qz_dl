from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import register_pluggable_module

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用装饰器注册模块之后即可通过注册表拿到以上结果

@register_pluggable_module(PluginType.MlpAtten)
class MLPAtten(nn.Module):
    def __init__(self, embed_dim):
        super(MLPAtten, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, embeddings):
        scores = self.attn_mlp(embeddings)
        attn_weights = F.softmax(scores, dim=1)
        weighted = embeddings * attn_weights
        return weighted


@register_pluggable_module(PluginType.PosEnc)
class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim):
        super(SinusoidalPE, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.embed_dim, "Embedding dimension mismatch."

        device = x.device
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))

        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)
        return x


@register_pluggable_module(PluginType.SA)
class SelfAtten(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAtten, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        return attended


@register_pluggable_module(PluginType.SelfPE)
class SelfAttenWithSinusoidalPE(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttenWithSinusoidalPE, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.embed_dim, "Embedding dimension mismatch."

        device = x.device
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))

        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        return attended
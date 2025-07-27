import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


def attention_block(embed_dim, atten_key: Optional[Literal["mlp", "self", "self-pe", "pe"]] = None):
    if atten_key is None:
        return nn.Identity()
    
    attens = {
        "mlp": MLPAttention(embed_dim=embed_dim), 
        "pe": SinusoidalPE(embed_dim=embed_dim),
        "self": SelfAttention(embed_dim=embed_dim),
        "self-pe": SelfAttentionWithSinusoidalPE (embed_dim=embed_dim)

    }
    return attens[atten_key]

    

class MLPAttention(nn.Module):
    def __init__(self, embed_dim):
        super(MLPAttention, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1) 
        )


    def forward(self, embeddings):
        # 计算每个 token 的注意力得分：[batch_size, seq_len, 1]
        scores = self.attn_mlp(embeddings)
        # 沿着序列维度归一化
        attn_weights = F.softmax(scores, dim=1)

        # 加权表示：[batch_size, seq_len, embed_dim]
        weighted = embeddings * attn_weights
        return weighted


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [B, T, D]
        Q = self.query(x)  # [B, T, D]
        K = self.key(x)    # [B, T, D]
        V = self.value(x)  # [B, T, D]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)                         # [B, T, T]

        attended = torch.matmul(attn_weights, V)  # [B, T, D]
        return attended


class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim):
        super(SinusoidalPE, self).__init__()
        self.embed_dim = embed_dim
        

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        assert D == self.embed_dim, "Embedding dimension mismatch."

        # 动态生成位置编码：[T, D]
        device = x.device
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        
        pe = torch.zeros(T, D, device=device)  # [T, D]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加入位置编码（broadcast 到 batch）
        x = x + pe.unsqueeze(0)  # [B, T, D]
        return x


class SelfAttentionWithSinusoidalPE(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionWithSinusoidalPE, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        assert D == self.embed_dim, "Embedding dimension mismatch."

        # 1. 动态生成位置编码：[T, D]
        device = x.device
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        
        pe = torch.zeros(T, D, device=device)  # [T, D]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 2. 加入位置编码（broadcast 到 batch）
        x = x + pe.unsqueeze(0)  # [B, T, D]

        # 3. QKV 投影
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 4. 注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)                  # [B, T, T]
        attended = torch.matmul(attn_weights, V)                           # [B, T, D]

        return attended
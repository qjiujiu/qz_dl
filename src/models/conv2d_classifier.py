from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block
from typing import Optional

import torch
import torch.nn as nn   
import torch.nn.functional as F


class Seq2ImageClassifier(nn.Module):
    def __init__(self, 
            vocab_size: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            output_dim: int, 
            dropout: float = 0.5,
            plugin_type: Optional[str] = None, 
        ):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.plugin = build_pluggable_block(plugin_type, embedding_dim)
        
        # in_channels=1 (输入实为一张灰度图)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # 先将输入通道 1 映射到 hidden_dim，以便相加
        self.residual_conv = nn.Conv2d(1, hidden_dim, kernel_size=1)

        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        # 自适应池化
        # 将任意大小的特征图 (H, W) 压缩为 (1, 1), 解决 max_len 依赖问题
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # MLP, 输入维度是最后一层卷积的 out_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, SeqLen]
        
        # Embedding & Plugin: [B, SeqLen, EmbedDim]
        embedded = self.embedding(x)
        embedded = self.plugin(embedded)

        #  Reshape to Image format: [B, SeqLen, EmbedDim] -> [B, 1, SeqLen, EmbedDim] (B, C, H, W)
        x = embedded.unsqueeze(1)
        residual = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Residual Connection
        out = out + residual 
        out = self.pool(out) 

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)

        # Global Pooling
        # 无论此时特征图长宽是多少，都压缩成 1x1, [B, Hidden*2, H', W'] -> [B, Hidden*2, 1, 1]
        out = self.global_pool(out)
        
        # Flatten & MLP
        out = out.view(out.size(0), -1) # [B, Hidden*2]
        out = self.dropout(out)
        
        return self.mlp(out)
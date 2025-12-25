from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block
import torch
import torch.nn as nn
from typing import Optional

 


class TextSeqClassifier(nn.Module):
    def __init__(self, 
            vocab_size: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            output_dim: int, 
            dropout: float = 0.5,
            plugin_type: Optional[PluginType] = None,
            **kwargs
        ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.plugin = build_pluggable_block(plugin_type, embedding_dim)

        # 一维卷积层堆叠
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 关键改进：使用自适应最大池化
        # 无论经过卷积后序列剩多长，强行池化成 output_size=1, output: [Batch, Channels, 1]
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # MLP 分类头, 输入维度等于 Block 2  out_channels
        fc_in_dim = hidden_dim * 2 
        
        self.mlp = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding x: [Batch, SeqLen] -> [Batch, SeqLen, EmbedDim]
        x = self.embedding(x)
        x = self.plugin(x)

        # Transpose for Conv1d [Batch, SeqLen, EmbedDim] -> [Batch, EmbedDim, SeqLen]
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)   # [Batch, Hidden*2, Reduced_SeqLen]
        x = self.adaptive_pool(x) # [Batch, Hidden*2, 1]
        x = x.squeeze(-1)         # [Batch, Hidden*2]

        # MLP
        x = self.dropout(x)
        logits = self.mlp(x)
        return logits




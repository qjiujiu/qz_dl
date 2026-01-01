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

    def forward(self, embedded: torch.Tensor, embed_perturbation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding x: [Batch, SeqLen] -> [Batch, SeqLen, EmbedDim]
        embedded = self.embedding(embedded)
        if embed_perturbation is not None:
            embedded = embedded + embed_perturbation


        embedded = self.plugin(embedded)

        # Transpose for Conv1d [Batch, SeqLen, EmbedDim] -> [Batch, EmbedDim, SeqLen]
        embedded = embedded.permute(0, 2, 1)

        embedded = self.conv_layers(embedded)   # [Batch, Hidden*2, Reduced_SeqLen]
        embedded = self.adaptive_pool(embedded) # [Batch, Hidden*2, 1]
        embedded = embedded.squeeze(-1)         # [Batch, Hidden*2]

        # MLP
        embedded = self.dropout(embedded)
        logits = self.mlp(embedded)
        return logits




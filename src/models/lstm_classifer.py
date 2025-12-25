from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block 

import torch
import torch.nn as nn
from typing import Optional


class LSTMSeqClassifier(nn.Module):  
    def __init__(self, 
            vocab_size: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            output_dim: int, 
            bidirectional: bool = True, 
            layers: int = 1,
            dropout: float = 0.3,
            plugin_type: Optional[PluginType] = None
        ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.plugin = build_pluggable_block(plugin_type, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 计算 LSTM 输出维度
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 构建分类头
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 输入数据 x: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)  
        embedded = self.plugin(embedded)
        
        # hidden: [num_layers * num_directions, batch, hidden_size]
        output, (hidden, cell) = self.lstm(embedded)
        pool_out = torch.max(output, dim=1)[0]
        
        # 改用最大池化, 避免过量padding污染 (不使用的Pack Padded Sequence, 若想使用数据集需要透传length)
        dropped = self.dropout(pool_out)
        logits = self.fc(dropped)
        return logits
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
        ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
         
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
        # 输入数据 x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # 使用 pack_padded_sequence 可以加速 LSTM 并且忽略 padding (可选优化)

        output, (hidden, cell) = self.lstm(embedded)
        
        # 取最后一个时间步的 hidden state
        # hidden: [num_layers * num_directions, batch, hidden_size]
        
        # 拼接正向和反向的最后一个 hidden state
        if self.lstm.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]
            
        dropped = self.dropout(final_hidden)
        logits = self.fc(dropped)
        return logits
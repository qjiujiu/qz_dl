import torch
import torch.nn as nn

from config.logger import logger
from models.classisifier import ClassifierBaseModel


class MalMLP(ClassifierBaseModel):
    def __init__(self, embedding_dim=128, hidden_dim=256, num_classes=8):
        super(MalMLP, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # logger.debug(f"{lstm_out.shape}, {cell.shape}, {cell.shape}")
        hidden_out = hidden[-1] + cell[-1]
        dropped_out = self.dropout(hidden_out)
        output = self.fc(dropped_out)
        return output
    
    def train_one_step(self, batch, **kwargs):
        return super().train_one_step(batch, **kwargs)
    
    def eval_one_step(self, batch, **kwargs):
        return super().eval_one_step(batch, **kwargs)
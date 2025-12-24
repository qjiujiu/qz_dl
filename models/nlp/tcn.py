import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nlp.atten import attention_block
from models.classisifier import ClassifierBaseModel

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, padding, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TCNTextClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, tcn_channels, output_dim, dropout=0.3, atten_key: str = None,):
        super(TCNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.atten = attention_block(embed_dim=embedding_dim, atten_key=atten_key)
        self.tcn = TCN(num_inputs=embedding_dim,
                       num_channels=tcn_channels,
                       kernel_size=3,
                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_channels[-1], output_dim)
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """ 嵌入层前向传播 """
        z = self.embedding(x) # [B, MAX_LEN, EMBEDDING_DIM]
        z = self.atten(z)
        return z  
    
    def forward(self, x):
        embedded = x.permute(0, 2, 1)  # [batch, embedding_dim, seq_len] for Conv1d

        tcn_out = self.tcn(embedded)  # [batch, channels, seq_len]
        tcn_last = tcn_out[:, :, -1]  # 取最后一个时间步的表示 [batch, channels]

        out = self.dropout(tcn_last)
        return self.fc(out)

    def train_one_step(self, batch, encoder = None, **kwargs):
        if encoder: 
            batch[0] = encoder(batch[0])
            return super().train_one_step(batch, **kwargs)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_ = self.forward(self.embed(x))
        return self.loss_fn(y_, y)

    def eval_one_step(self, batch, encoder = None, **kwargs):
        if encoder:
            batch[0] = encoder(batch[0])
            return super().eval_one_step(batch, **kwargs)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_ = self.forward(self.embed(x))
        return y_
    
    def evalution(self, dataloader, **kwargs):
        return super().evalution(dataloader, **kwargs) 
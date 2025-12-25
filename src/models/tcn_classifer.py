from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block 
from typing import Optional, List
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # TCN标准实践: 使用 weight_norm, 有助于收敛
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # TCN 中通常会修剪掉多余的 padding (Chomp1d)，
        # 但为了文本分类任务保留上下文，这里使用常规 Padding 保持长度不变即可
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels = n_outputs,
                out_channels = n_outputs,
                kernel_size = kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
                                 
        # 1x1 卷积用于调整残差连接的维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        #  TCN 卷积核尺寸必须是奇数, 确保非因果 padding 对齐
        super(TCN, self).__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd (3, 5, 7...) to maintain dimension alignment with padding.")
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 这种 padding 计算方式能保证输出长度 = 输入长度 (Same Padding)
            # 前提是 kernel_size 是奇数，或者使用特定的 padding 策略
            padding = (kernel_size - 1) * dilation_size // 2
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    


class TCNSeqClassifier(nn.Module):
    def __init__(self, 
            vocab_size: int , 
            embedding_dim: int, 
            tcn_channels: List[int], 
            output_dim: int,
            dropout=0.3,
            plugin_type: Optional[PluginType] = None
        ):
        super(TCNSeqClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.plugin = build_pluggable_block(plugin_type, embedding_dim)
        
        self.tcn = TCN(
            num_inputs=embedding_dim,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # TCN 最终输出通道数是 tcn_channels 最后一个元素
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_channels[-1], output_dim)
    
    
    def forward(self, x):
        # Embedding -> [batch, seq_len, embed_dim]
        x = self.embedding(x) 
        
        # Plugin (在转置之前进行，因为 Attention 通常处理 seq_len 维度)
        x = self.plugin(x)     

        # Conv1d 需要通道在中间: [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        tcn_out = self.tcn(x)         # [batch, channels, seq_len]
        out = torch.max(tcn_out, dim=2)[0]

        # 策略 A: 取最后一个时间步 (适合因果卷积/序列建模, RNN/LSTM标准操作, 但是此处使用这个策略可能会被padding影响)
        # 策略 B: 全局最大池化 (Global Max Pooling, 适合关键词分类, 采用这种方法可能会有更好的效果)
                
        out = self.dropout(out)
        return self.fc(out)

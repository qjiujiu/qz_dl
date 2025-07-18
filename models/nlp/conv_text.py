import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classisifier import ClassifierBaseModel

class Conv2dTextClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len, **kwargs):
        super(Conv2dTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 二维卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # 残差连接 1x1 卷积（输入通道=1，输出通道=hidden_dim）
        self.residual_conv = nn.Conv2d(1, hidden_dim, kernel_size=1, stride=1)

        # 展平后输入给 MLP 的张量的维度，模型结构改动之后记得修改这里！！
        flatten_dim = hidden_dim * max_len * embedding_dim // 16

        # MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """ 嵌入层前向传播 """
        return self.embedding(x)  # [B, MAX_LEN, EMBEDDING_DIM]

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        - 输入: [B, MAX_LEN, EMBEDDING_DIM]
        - 输出: [B, output_dim]
        """
        B, MAX_LEN, EMBEDDING_DIM = embedded.shape
        embedded = embedded.unsqueeze(1)  # [B, 1, MAX_LEN, EMBEDDING_DIM]

        # 卷积 + 池化
        x = self.conv1(embedded)
        x = torch.relu(x)
        x = self.pool(x)

         # 残差分支（使用 1x1 卷积 + 池化）
        residual = self.residual_conv(embedded)
        residual = self.pool(residual)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 匹配残差尺寸
        if residual.shape != x.shape:
            residual = F.adaptive_avg_pool2d(residual, output_size=x.shape[-2:])

        # 加上残差连接
        x += residual
        x = self.dropout(x)

        # 展平
        x = x.view(B, -1)   # 扁平化
        # print(x.shape) 
        output = self.mlp(x)

        return output

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


class Conv1dTextClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, **kwargs):
        super(Conv1dTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 一维卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(0.5)

        # 使用 MLP 作为分类头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),     # 第一层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),            # 第二层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)       # 输出层
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """ 嵌入层前向传播 """
        return self.embedding(x)  # [B, MAX_LEN, EMBEDDING_DIM]

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        - 输入: [B, MAX_LEN, EMBEDDING_DIM]
        - 输出: [B, output_dim]
        """

        # 转换为 [B, EMBEDDING_DIM, MAX_LEN]
        embedded = embedded.transpose(1, 2)   # [B, E, L]
        # print(embedded.shape)
        
        # 卷积 + 池化
        x = self.conv(embedded)
        # print(x.shape)

        x = x.mean(dim=2)                     # [B, E]
        # print(x.shape)

        # 使用 MLP 作为分类头
        x = self.dropout(x)
        output = self.mlp(x)
        return output
    
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
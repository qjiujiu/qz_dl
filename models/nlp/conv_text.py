import torch
import torch.nn as nn


from torch import Tensor
from models.classisifier import ClassifierBaseModel


class Conv2dTextClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, **kwargs):
        super(Conv2dTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 确保嵌入矩阵可以reshape为正方形图像
        assert embedding_dim == image_size * image_size, "Embedding dimension must match the square of image size"
        image_size =  int(embedding_dim ** 0.5)

        # 二维卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * (image_size // 4) ** 2, 512),
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
        embedded = embedded.view(B, MAX_LEN, self.image_size, self.image_size) 

        # 卷积 + 池化
        x = self.conv1(embedded)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)  # [B, hidden_dim * (image_size // 4) ** 2]

        # 使用 MLP 作为分类头
        output = self.mlp(x)
        return output




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
        print(embedded.shape)
        
        # 卷积 + 池化
        x = self.conv(embedded)
        print(x.shape)

        x = x.mean(dim=2)                     # [B, E]
        print(x.shape)

        # 使用 MLP 作为分类头
        x = self.dropout(x)
        output = self.mlp(x)
        return output
    
    def train_one_step(self, batch, only_emebedding = False):
            if only_emebedding: 
                return super().train_one_step(batch, only_emebedding)
            
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_ = self.forward(self.embed(x))
            return self.loss_fn(y_, y)
        
    def eval_one_step(self, batch, only_emebedding = False):
        if only_emebedding:
            return super().eval_one_step(batch, only_emebedding)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_ = self.forward(self.embed(x))
        return y_
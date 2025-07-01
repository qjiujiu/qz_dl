# models/nlp/lstm_text_classifier.py
import torch.nn as nn
from torch import Tensor
from models.classisifier import ClassifierBaseModel


class LSTMTextClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len, pretrained_embeddings=None):
        """ 初始化 LSTM 文本分类模型
        参数：
            - vocab_size: 词汇表大小
            - embedding_dim: 嵌入层维度
            - hidden_dim: LSTM 隐藏层维度
            - output_dim: 输出层维度（类别数）
            - max_len: 输入文本的最大长度
            - pretrained_embeddings: 预训练词向量（可选）
        """
        super(LSTMTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def embed(self, x: Tensor):
        """ 嵌入层前向传播
            - 输入：[batch_size, max_len] 的 token 索引张量
            - 输出：[batch_size, max_len, embedding_dim] 的嵌入表示
        """
        return self.embedding(x)

    def forward(self, embedded):
        """ 从嵌入开始的前向传播（跳过嵌入层）
        输入：
            - 输入: [batch_size, max_len]    token 索引张量
            - 输出: [batch_size, output_dim] logits
        """
        # 获取 LSTM 最后一层的隐状态（最后一个时间步的输出）
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_out = hidden[-1]
        dropped_out = self.dropout(hidden_out)
        output = self.fc(dropped_out)
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
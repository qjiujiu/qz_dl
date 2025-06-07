# models/nlp/lstm_text_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len, pretrained_embeddings=None):
        """
        初始化 LSTM 文本分类模型
        
        参数：
        - vocab_size: 词汇表大小
        - embedding_dim: 嵌入层维度
        - hidden_dim: LSTM 隐藏层维度
        - output_dim: 输出层维度（类别数）
        - max_len: 输入文本的最大长度
        - pretrained_embeddings: 预训练词向量（可选）
        """
        super(LSTMTextClassifier, self).__init__()
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            # 如果提供预训练的词向量，使用它
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        前向传播函数
        
        参数：
        - x: 输入文本的索引表示，形状为 [batch_size, max_len]
        
        返回：
        - 输出层的分类结果
        """
        # 嵌入层，获取词向量
        embedded = self.embedding(x)
        
        # LSTM 层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 获取 LSTM 最后一层的隐状态（最后一个时间步的输出）
        hidden_out = hidden[-1]
        
        # Dropout 层
        dropped_out = self.dropout(hidden_out)
        
        # 全连接层输出
        output = self.fc(dropped_out)
        
        return output
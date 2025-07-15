import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classisifier import ClassifierBaseModel

from config.metrics.metrics_res_template import ClassificationResult
from config.datasets.dataset_instance.mal_api import MalAPITextDataset
from config.logger import logger

from contextlib import contextmanager
from torch import Tensor
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


@contextmanager
def disable_dropout(model):
    """临时禁用模型 Dropout """
    original_p = model.dropout.p 
    try:
        model.dropout.p = 0.0
        yield  
    finally:
        model.dropout.p = original_p 


class LSTMTextAdvClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional = False, layers = 0, **kwargs):
        """ 初始化 LSTM 文本分类模型
        参数：
            - vocab_size: 词汇表大小
            - embedding_dim: 嵌入层维度
            - hidden_dim: LSTM 隐藏层维度
            - output_dim: 输出层维度（类别数）
            - max_len: 输入文本的最大长度
            - pretrained_embeddings: 预训练词向量（可选）
        """
        super(LSTMTextAdvClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 引入layernorm与否并没有造成什么性能影响
        # self.embed_ln = nn.LayerNorm(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = bidirectional)
        self.dropout = nn.Dropout(0.5)
        self.fc = self._build_fc(hidden_dim, output_dim, layers=layers)

    def embed(self, x: Tensor):
        """ 嵌入层前向传播
            - 输入：[batch_size, max_len] 的 token 索引张量
            - 输出：[batch_size, max_len, embedding_dim] 的嵌入表示
        """
        z = self.embedding(x)
        # z = self.embed_ln(z)
        return z

    def forward(self, embedded):
        """ 从嵌入开始的前向传播（跳过嵌入层）
        输入：
            - 输入: [batch_size, max_len]    token 索引张量
            - 输出: [batch_size, output_dim] logits
        """
        # 获取 LSTM 最后一层的隐状态（最后一个时间步的输出）
        # embedded.shape: torch.Size([8, 200, 128])
        # hidden.shape: torch.Size([1, 8, 256])
        # hidden_out.shape: torch.Size([8, 256])
        # dropped_out.shape: torch.Size([8, 256])
        # output.shape: torch.Size([8, 8])
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # [batch, 200, 256], [1, 8, 256], [1, 8, 256] 单向lstm
        # [batch, 200, 512], [2, 8, 256], [2, 8, 256] 假设开启的双向lstm

        dropped_out = self.dropout(hidden[0] + hidden[1])
        output = self.fc(dropped_out)
        return output
    
    def _build_fc(self, in_dims: int, out_dims: int, layers: int = 0) -> nn.Sequential:
        """ 构建全连接层：
            - 头部层：in_dims -> hidden_dims
            - 中间层：hidden_dims ->  hidden_dims
            - 末尾层：hidden_dims ->  out_dims
        """
        fc_layers = nn.Sequential()
        
        # 设置隐藏层维度为输出维度（我们令所有中间层保持相同维度）
        hidden_dim = in_dims
        for i in range(layers):
            if i == 0:
                fc_layers.add_module(f"fc_{i}", nn.Linear(in_dims, hidden_dim))
            else:
                fc_layers.add_module(f"fc_{i}", nn.Linear(hidden_dim, hidden_dim))
            
            # 添加 ReLU 和 batch-norm
            fc_layers.add_module(f"bn_{i}", nn.BatchNorm1d(hidden_dim))
            fc_layers.add_module(f"relu_{i}", nn.ReLU())
        
        # 最后一层输出
        fc_layers.add_module("final_fc", nn.Linear(hidden_dim, out_dims))
        return fc_layers
        

    def train_one_step(self, batch, encoder = None, adv_type="fgsm"):
        if encoder: 
            batch[0] = encoder(batch[0])
            return super().train_one_step(batch)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        adv_embed = self.embed(x)

        if adv_type == "fgsm":
            adv_embed = self.fgsm_attack(adv_embed, y)
        elif adv_type == "pgd":
            adv_embed = self.pgd_attack(adv_embed, y)

        y_ = self.forward(adv_embed)
        return self.loss_fn(y_, y)
    
    def eval_one_step(self, batch, encoder = None, adv_type="fgsm"):
        if encoder:
            batch[0] = encoder(batch[0])
            return super().eval_one_step(batch)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        adv_embed = self.embed(x)
        if adv_type == "fgsm":
            adv_embed = self.fgsm_attack(adv_embed, y)
        elif adv_type == "pgd":
            adv_embed = self.pgd_attack(adv_embed, y)
        
        self.eval() # 完成对抗向量的构造之后再切回 eval 模式
        y_ = self.forward(adv_embed)
        return y_
    

    def evalution(self, dataloader, **kwargs):
        """评估模型并返回性能指标"""
        self.eval()
        all_preds, all_reals = [], []

        with tqdm(dataloader, desc="Evaluating", unit="batch") as tepoch:
            for batch in tepoch:
                outputs = self.eval_one_step(batch, **kwargs)
                y_ = torch.argmax(outputs, dim=1)

                _, y  = batch
                all_reals.extend(y.cpu().numpy())
                all_preds.extend(y_.cpu().numpy())

        # 计算各项指标
        accuracy = accuracy_score(all_reals, all_preds)
        precision = precision_score(all_reals, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_reals, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_reals, all_preds, average='macro', zero_division=0)
        
        return ClassificationResult(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
    
    def pgd_attack(self, embed, labels, epsilon = 0.1, alpha=0.01, iters = 5):
        with disable_dropout(self):
            adv = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)  
            
            self.train()
            for _ in range(iters):
                self.zero_grad()
                output = self.forward(adv)

                loss = F.cross_entropy(output, labels)
                loss.backward()

                adv = adv + alpha * adv.grad.sign()
                eta = torch.clamp(adv - embed, -epsilon, epsilon)
                adv = torch.clamp(embed + eta, 0, 1).detach_().requires_grad_(True)  # 更新对抗样本

        return adv.detach()
    
    def fgsm_attack(self, embed, labels, epsilon = 0.1):
        with disable_dropout(self):
            embed = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)
        
            self.train()
            output = self.forward(embed)
            loss = F.cross_entropy(output, labels)
            loss.backward()
    
            adv_emb = embed + epsilon * embed.grad.sign()  # 依据梯度更新对抗样本

        return adv_emb.detach()
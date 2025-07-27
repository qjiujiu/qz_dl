import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classisifier import ClassifierBaseModel
from models.nlp.atten import attention_block

from config.metrics.metrics_res_template import ClassificationResult
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
            bidirectional: bool = False, layers: int = 0, atten_key: str = None,
            **kwargs
        ):
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
        self.atten = attention_block(embed_dim=embedding_dim, atten_key=atten_key)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = bidirectional)
        self.dropout = nn.Dropout(0.5)
        self.fc = self._build_fc(hidden_dim, output_dim, layers=layers)

    def embed(self, x: Tensor):
        """ 嵌入层前向传播
            - 输入：[batch_size, max_len] 的 token 索引张量
            - 输出：[batch_size, max_len, embedding_dim] 的嵌入表示
        """
        z = self.embedding(x)
        z = self.atten(z)
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
        

    def train_one_step(self, batch, encoder = None, adv_type="fgsm", pgd_iters = 3):
        if encoder: 
            batch[0] = encoder(batch[0])
            return super().train_one_step(batch)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        adv_embed = self.embed(x)

        attacker = {
            "fgsm": self.fgsm_attack, 
            "pgd": self.pgd_attack, 
            "l2-gaus": self.l2_gaussian_attack,
            "linf-gaus": self.linf_gaussian_attack
        }[adv_type]
        
        adv_embed = attacker(adv_embed, y, iters = pgd_iters)


        y_ = self.forward(adv_embed)
        return self.loss_fn(y_, y)
    
    def eval_one_step(self, batch, encoder = None, adv_type="fgsm", pgd_iters = 3):
        if encoder:
            batch[0] = encoder(batch[0])
            return super().eval_one_step(batch)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        adv_embed = self.embed(x)
        attacker = {
            "fgsm": self.fgsm_attack, 
            "pgd": self.pgd_attack, 
            "l2-gaus": self.l2_gaussian_attack,
            "linf-gaus": self.linf_gaussian_attack
        }[adv_type]
        
        adv_embed = attacker(adv_embed, y, iters = pgd_iters)
        
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
    
    def pgd_attack(self, embed, labels, epsilon = 0.1, alpha=0.01, iters = 3):
        with disable_dropout(self):
            adv = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)  
            
            self.train()
            for _ in range(iters):
                output = self.forward(adv)
                loss = F.cross_entropy(output, labels)
                loss.backward()

                adv = adv + alpha * adv.grad.sign()
                eta = torch.clamp(adv - embed, -epsilon, epsilon)
                adv = torch.clamp(embed + eta, 0, 1).detach_().requires_grad_(True)  # 更新对抗样本

        return adv.detach()
    
    def fgsm_attack(self, embed, labels, epsilon = 0.1, iters=3):
        with disable_dropout(self):
            embed = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)
        
            self.train()
            output = self.forward(embed)
            loss = F.cross_entropy(output, labels)
            loss.backward()
    
            adv_emb = embed + epsilon * embed.grad.sign()  # 依据梯度更新对抗样本

        return adv_emb.detach()

    def l2_gaussian_attack(self, embed, labels, epsilon=0.1, iters=3):
        """  epsilon 范围之中添加高斯噪声，用于对比 FGSM 攻击
        """
        with disable_dropout(self):
            embed = embed.clone().detach().to(self.device)
            batch_size, seq_len, dim = embed.shape

            # 生成标准正态分布的噪声
            noise = torch.randn_like(embed)

            # 计算每个样本的 L2 范数，并进行归一化
            noise_norm = torch.norm(noise.view(batch_size, -1), dim=1, keepdim=True)  # shape: [batch, 1]
            noise_unit = noise / noise_norm.view(batch_size, 1, 1)  # 归一化到单位向量

            # 缩放到 epsilon 范围内
            adv_emb = embed + epsilon * noise_unit

        return adv_emb.detach()
    
    def linf_gaussian_attack(self, embed, labels, epsilon=0.1, iters=3):
        """  L-inf 范数约束之中添加 clip 处理高斯噪声，每个元素幅度不超过 [-eps, +eps]
        """
        with disable_dropout(self):
            embed = embed.clone().detach().to(self.device)
            
            # 生成标准高斯噪声
            noise = torch.randn_like(embed)

            # Clip 每个元素使其不超过 ± epsilon
            noise = torch.clamp(noise, min=-epsilon, max=epsilon)

            adv_emb = embed + noise

        return adv_emb.detach()
    

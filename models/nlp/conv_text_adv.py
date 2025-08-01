import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classisifier import ClassifierBaseModel

from config.metrics.metrics_res_template import ClassificationResult
from config.datasets.dataset_instance.mal_api import MalAPITextDataset
from config.logger import logger
from models.nlp.atten import attention_block
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

class Conv2dTextAdvClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, atten_key: str = None, **kwargs):
        super(Conv2dTextAdvClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.atten = attention_block(embed_dim=embedding_dim, atten_key=atten_key)
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
        z = self.embedding(x)  # [B, MAX_LEN, EMBEDDING_DIM]
        z = self.atten(z)
        return z

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



class Conv1dTextAdvClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, atten_key: str = None, **kwargs):
        super(Conv1dTextAdvClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.atten = attention_block(embed_dim=embedding_dim, atten_key=atten_key)

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
        z = self.embedding(x)  # [B, MAX_LEN, EMBEDDING_DIM]
        z = self.atten(z)
        return z

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

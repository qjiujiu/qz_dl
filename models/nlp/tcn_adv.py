import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classisifier import ClassifierBaseModel
from config.metrics.metrics_res_template import ClassificationResult
from config.logger import logger
from contextlib import contextmanager
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



class TCNTextAdvClassifier(ClassifierBaseModel):
    def __init__(self, vocab_size, embedding_dim, tcn_channels, output_dim, dropout=0.3):
        super(TCNTextAdvClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.tcn = TCN(num_inputs=embedding_dim,
                       num_channels=tcn_channels,
                       kernel_size=3,
                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_channels[-1], output_dim)

    def forward(self, x):
        embedded = self.embedding(x)          # [batch, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch, embedding_dim, seq_len] for Conv1d

        tcn_out = self.tcn(embedded)  # [batch, channels, seq_len]
        tcn_last = tcn_out[:, :, -1]  # 取最后一个时间步的表示 [batch, channels]

        out = self.dropout(tcn_last)
        return self.fc(out)

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
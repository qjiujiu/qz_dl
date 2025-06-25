import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

from config import logger 
from collections import namedtuple

# 准确率、精确率、召回率、F1 分数、混淆矩阵
EvalResult = namedtuple("EvalResult", ["acc", "prec", "recall", "f1", "cm"])



class ClassifierCVModelEngine:
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device = None):
        """封装分类模型的训练、评估、攻击功能"""
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        if torch.cuda.is_available():
            self.device = device or torch.device("cuda")
        else:
            self.device = device or torch.device("cpu")

        self.model.to(self.device)

    def train(self, dataloader: DataLoader, epochs: int = 1):
        """训练模型"""
        self.model.train()
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            with tqdm(dataloader, desc=f"[Train] Epoch {ep}/{epochs}", unit="batch") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / (pbar.n + 1))

            avg_loss = total_loss / len(dataloader)
            logger.info(f"[Epoch {ep}/{epochs}] Avg Loss = {avg_loss:.4f}")


    def evaluate(self, dataloader: DataLoader) -> dict:
        """评估模型，输出 acc / precision / recall / f1 分数等指标"""
        self.model.eval()
        all_preds, all_reals = [], []
        with torch.no_grad(), tqdm(dataloader, desc="[Eval]", unit="batch") as pbar:
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                y_ = logits.argmax(dim=1)
                
                all_reals.extend(y.cpu().numpy())
                all_preds.extend(y_.cpu().numpy())

                acc = (np.array(all_preds) == np.array(all_reals)).mean()
                pbar.set_postfix(acc=acc * 100)

        # 构建混淆矩阵 + 精确率、召回率、F1
        cm = confusion_matrix(all_reals, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_reals, all_preds, average='macro')
        acc = np.mean(np.array(all_preds) == np.array(all_reals))

        logger.info(f"[Eval] Accuracy: {acc * 100:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

        return EvalResult(
            acc=acc,
            prec=precision,
            recall=recall,
            f1=f1,
            cm=cm
        )

    def fgsm_attack(self, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
        """FGSM 对抗样本生成"""
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)

        logits = self.model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        x_adv = x_adv + epsilon * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()


    def pgd_attack(self, x: torch.Tensor, y: torch.Tensor, epsilon=0.3, alpha=0.01, iters=5) -> torch.Tensor:
        """PGD 对抗样本生成"""
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        ori_x = x.clone().detach()
        x_adv = x.clone().detach().requires_grad_(True)

        
        for _ in range(iters):
            self.model.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            
            x_adv = x_adv + alpha * x_adv.grad.sign()
            eta = torch.clamp(x_adv - ori_x, -epsilon, epsilon)
            x_adv = torch.clamp(ori_x + eta, 0, 1).detach_().requires_grad_(True)

        return x_adv.detach()

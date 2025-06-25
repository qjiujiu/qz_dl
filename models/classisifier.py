import torch
import torch.nn as nn
import torch.optim as optim

import os
import uuid, json

from tqdm import tqdm
from datetime import datetime

from abc import ABC, abstractmethod
from config.logger import logger
from config.params_parser import CommonCfgParams

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


class ClassifierBaseModel(ABC, nn.Module):
    def __init__(self, log_dir="logs", *args, **kwargs):
        """ 初始化 BaseModel 类

        参数：
        - loss_fn: 损失函数
        - optimizer: 优化器
        """
        super(ClassifierBaseModel, self).__init__()

        self.ctx = None
        self.loss_fn = None
        self.optimizer = None
        self.device = None
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

        # 初始化历史记录，并且记录训练配置
        self.history = {
            'train_loss': [],
            'val_metrics': []
        }
        

    def _generate_log_id(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        unique_id = str(uuid.uuid4())[:8]
        self.log_id = f"{timestamp}-{unique_id}"
        return self.log_id


    def _generate_log(self):
        """ 本次训练的所有信息保存到 logs/log_id.json """
        config = {
            'loss_fn': str(self.loss_fn.__class__.__name__),
            'optimizer': str(self.optimizer.__class__.__name__),
            'config': self.ctx.as_dict(),  
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        model_summary = str(self)  # 可选：输出模型结构
        # TODO 增加一个权重缓存的位置
        log_data = {
            "log_id": self.log_id,
            "history": self.history,
            "config": config,
            "model_summary": model_summary
        }

        log_path = os.path.join(self.log_dir, f"{self.log_id}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        logger.debug(f"Training log saved to: {log_path}")


    @abstractmethod
    def forward(self, *args, **kwargs):
        """ 每个子类需要实现自己的前向传播逻辑
        """
        pass


    def setup_ctx(self,  ctx: CommonCfgParams):
        self.ctx = ctx
        self.device = ctx.device
        return self.to(self.device)
    
    def setup_loss(self, loss_fn):
        self.loss_fn = loss_fn
        return self

    def setup_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self

    

    
    # 绝大部分情况之下，eval_one_step/train_one_step 推理操作等同于forward 操作
    @abstractmethod 
    def eval_one_step(self, batch, **kwargs):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_ = self(x)
        return y_

    @abstractmethod
    def train_one_step(self, batch, **kwargs):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_ = self(x)
        return self.loss_fn(y_, y)


    def evalution(self, dataloader, **kwargs):
        """评估模型并返回性能指标"""
        self.eval()
        all_preds, all_reals = [], []

        with torch.no_grad(), tqdm(dataloader, desc="Evaluating", unit="batch") as tepoch:
            for batch in tepoch:
                outputs = self.eval_one_step(batch)
                y_ = torch.argmax(outputs, dim=1)

                _, y  = batch
                all_reals.extend(y.cpu().numpy())
                all_preds.extend(y_.cpu().numpy())

        # 计算各项指标
        accuracy = accuracy_score(all_reals, all_preds)
        precision = precision_score(all_reals, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_reals, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_reals, all_preds, average='macro', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics
    

    def train_one_epoch(self, dataloader, val_loader=None, **kwargs):
        """ 训练一个 epoch，这个方法包含默认版本，子类可以重写，也可以直接复用
        """
        self.train() 
        total_loss, metrics = 0, {}
        with tqdm(dataloader, desc="Training", unit="batch") as tepoch:
            for batch in tepoch:
                loss = self.train_one_step(batch, **kwargs)
                loss.backward()

                self.optimizer.step()       # 更新模型参数
                self.optimizer.zero_grad()  # 清除梯度
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))  # 更新进度条的损失
            
            if val_loader is not None:
                metrics = self.evalution(val_loader)
                
        total_loss /= len(dataloader)
        return total_loss, metrics


    def train_multiple_epochs(self, loader,val_loader = None, epochs = 10, **kwargs):
        """ 训练多个epoch
        """
        self._generate_log_id()
        logger.debug(f"当前轮训练log-id: {self.log_id} 已开启...")

        for epoch in range(epochs):
            avg_loss, metrics = self.train_one_epoch(loader, val_loader, **kwargs)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            self.history['train_loss'].append(avg_loss)
            self.history['val_metrics'].append(metrics)

        self._generate_log()
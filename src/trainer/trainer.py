import torch
import torch.nn as nn
import os
import json
import uuid
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

from src.utils.logx import logger
from src.utils.dumps import to_jsonable, ensure_dirs
from src.schemas.context import ExpContext
from src.evaluation import EvalState
from src.trainer.build_helper import build_optimizer, build_scheduler, build_loss_fn


class Trainer:
    def __init__(self, context: ExpContext, model: nn.Module, train_loader, val_loader=None):
        self.ctx = context
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 硬件设置
        self.device = self.ctx.train_config.torch_device
        self.model.to(self.device)
        
        # 优化器、调度器、损失函数
        self.optimizer = build_optimizer(self.ctx.train_config, self.model)
        self.scheduler = build_scheduler(self.ctx.train_config, self.optimizer)
        self.loss_fn = build_loss_fn(self.ctx.train_config)
        
        # 日志管理
        self.task_id =  self.ctx.task_id
        self.history = {
            'train_loss': [], 
            'val_metrics': []
        }
        
        # 确保目录存在
        ensure_dirs(
            self.ctx.model_config.checkpoint_dir, 
            self.ctx.model_config.ouputs_trace_dir,
            self.ctx.model_config.logs_dir
        )
        
        
    def train(self):
        epochs = self.ctx.train_config.epochs
        logger.debug(f"Starting training task: {self.task_id} on {self.device}, epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            
            total_loss = 0            
            with tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # 对抗训练配置在根目录下
                    if self.ctx.adv_config.enable:
                        pass

                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_loss)
            
            # === Validation Loop ===
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.evaluate()
                self.history['val_metrics'].append(val_metrics)
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                logger.info(f"Epoch {epoch} Val: {metrics_str}")

            # === Scheduler Step ===
            if self.scheduler:
                self.scheduler.step()
                
            # === Save Checkpoint ===
            self._save_checkpoint(epoch, val_metrics)

        self._save_final_log()

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
        state = EvalState(
            labels=np.concatenate(all_labels),
            predicts=np.concatenate(all_preds)
        )
        
        return {
            "acc": state.accuracy, 
            "f1_macro": state.macro_f1_score,
            "precision_macro": state.macro_precision
        }

    def _save_checkpoint(self, epoch: int, metrics: dict):
        name = self.ctx.model_config.name
        ckpt_dir = self.ctx.model_config.checkpoint_dir
        
        save_path = ckpt_dir / f"{name}_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), save_path)
        logger.debug(f"Saved checkpoint to {save_path}")

    def _save_final_log(self):
        trace = to_jsonable({
            "task_id": self.task_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.ctx.model_dump(mode='json'),
            "history": self.history
        })
        
        log_path = self.ctx.model_config.ouputs_trace_dir / f"{self.task_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=4, ensure_ascii=False)
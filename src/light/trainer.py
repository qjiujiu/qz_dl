from src.utils.logx import logger
from src.utils.dumps import dump_to_json, ensure_dirs
from src.schemas.context import ExpContext
from src.evaluation import EvalState
from src.light.build_helper import build_optimizer, build_scheduler, build_loss_fn

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Dict


class Trainer:
    def __init__(self, ctx: ExpContext, model: nn.Module, train_loader, val_loader=None):
        self.ctx: ExpContext = ctx
        self.model: nn.Module = model
        
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
            self.ctx.network_config.checkpoint_dir, 
            self.ctx.network_config.ouputs_trace_dir,
            self.ctx.network_config.ouputs_log_dir,
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
            val_metrics = self.evaluate()
            
            self.history['train_loss'].append(avg_loss)
            self.history['val_metrics'].append(val_metrics)

            logger.info(f"Epoch {epoch},  avg_loss: {avg_loss:.4f}, val_metrics: {val_metrics}")

            if self.scheduler:
                self.scheduler.step()
                
        self._save_checkpoint(suffix=f"epoch-{epochs}")
        self._save_trace()

    @torch.no_grad()
    def evaluate(self) -> Dict:
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
        return state.calculate()

    def _save_checkpoint(self, suffix: str):
        name = self.ctx.network_config.name
        ckpt_dir = self.ctx.network_config.checkpoint_dir
        
        save_path = ckpt_dir / f"{name}-{suffix}.pth"
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def _save_trace(self):
        trace = {
            "task_id": self.task_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.ctx.model_dump(mode='json'),
            "history": self.history
        }
        
        trace_path = self.ctx.network_config.ouputs_trace_dir / f"{self.task_id}.json"
        dump_to_json(trace, trace_path)

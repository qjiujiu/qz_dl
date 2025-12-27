from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.evaluation import EvalState
from src.light.trainer import Trainer 

from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
from typing import Dict
import torch
import torch.nn as nn
import numpy as np

# 专门用于训练 Bert-base 下游任务分类模型

class BertTrainer(Trainer):
    """
    专门适配 HuggingFace Transformers (BERT) 的训练器
    继承自通用 Trainer，重写了模型加载、优化器构建、训练循环和评估逻辑。
    """
    def __init__(self, ctx: ExpContext, model: nn.Module, train_loader, val_loader=None):
        super().__init__(ctx=ctx, model=model, train_loader=train_loader, val_loader=val_loader)
        
        # 构建 Warmup 调度器, 计算总步数, 然后采用 10% Warmup
        epochs = ctx.train_config.epochs
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * 0.1) 
        
        logger.info(f"BertTrainer Initialized. Warmup Steps: {num_warmup_steps}, Total Steps: {total_steps}")
        
        # 覆盖原先的优化器
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
        

    def train(self):
        epochs = self.ctx.train_config.epochs
        logger.debug(f"Starting BERT training task: {self.task_id} on {self.device}, epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for batch in pbar:
                    # BERT 里面每个 batch 通常是字典，包含 input_ids, attention_mask, labels
                    # 自动把字典里的 tensor 移到 GPU
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    if self.ctx.adv_config.enable:
                        pass

                    self.optimizer.zero_grad()
                    
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward
                    loss.backward()
                    
                    # 梯度裁剪 (BERT 特有)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step() # 每个 step 都更新 learning rate

                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            
            # 评估
            val_metrics = self.evaluate()
            
            # 记录历史
            self.history['train_loss'].append(avg_loss)
            self.history['val_metrics'].append(val_metrics)

            # 打印日志
            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | {str(val_metrics)}")


        self._save_checkpoint(suffix=f"epoch-{epochs}-{self.ctx.task_id[:6]}")
        self._save_trace()

    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 使用 EvalState 统一计算指标
        state = EvalState(
            labels=np.concatenate(all_labels),
            predicts=np.concatenate(all_preds)
        )
        return state.calculate()
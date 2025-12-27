from src.utils.logx import logger
from src.datasets.instances import malapibert 
from src.schemas.context import ExpContext 
from src.configs.bert import ctx


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report



def run_training(ctx: ExpContext):
    device = ctx.train_config.device
    logger.info("Loading Datasets...")
    train_loader, val_loader = malapibert.build_datamodule(ctx)
    
    logger.info(f"Loading Pretrained Model: microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='microsoft/codebert-base', 
        num_labels=ctx.network_config.num_classes,
        weights_only=False,
    )
    model.to(device)


    # 优化器设置
    optimizer = optim.AdamW(
        params=model.parameters(), 
        lr=ctx.train_config.lr, 
        weight_decay=ctx.train_config.weight_decay,
    )
    
    # Warmup 步数通常设为总步数的 10%
    epochs = ctx.train_config.epochs
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = total_steps
    )

    logger.info(f"Start Training: Epochs={epochs}, BatchSize={ctx.train_config.batch_size}, LR={ctx.train_config.lr}")

    # --- 辅助函数：训练一个 Epoch ---
    def train_epoch(epoch_idx):
        model.train()
        total_loss = 0
        all_preds, all_labels= [], []
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch_idx+1}/{epochs}")
        for batch in pbar:
            input_ids, attention_mask, labels = (    
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device), 
                batch["labels"].to(device),
            )
            
            # 梯度清零, 然后执行前向传播
            model.zero_grad()
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            # 反向传播 + 梯度裁剪 (防止梯度爆炸，Transformers 训练常用技巧)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()
            scheduler.step()

            # 记录统计信息
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    # --- 辅助函数：验证 ---
    def eval_epoch(epoch_idx):
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        logger.info(f"Epoch: {epoch_idx}, Evaluating...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attention_mask, labels = (    
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device), 
                    batch["labels"].to(device),
                )

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        
        # 计算详细指标
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, acc, macro_f1

    # --- 主循环 ---
    best_f1 = 0.0
    save_dir = Path(ctx.network_config.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(epoch)
        logger.info(f"Epoch {epoch+1} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, val_f1 = eval_epoch(epoch)
        logger.info(f"Epoch {epoch+1} Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # 保存最佳模型 (根据 F1 Score)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = save_dir / "best_model.pt"
            logger.info(f"New Best F1: {best_f1:.4f}. Saving model to {save_path}")
            
            torch.save(model.state_dict(), save_path)
            
    logger.info("Training Finished.")
    

if __name__ == "__main__":
    run_training(ctx)
    
    
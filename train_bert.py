import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report


from src.utils.cuda import select_gpu
from src.utils.logx import logger
from src.datasets.instances import malapibert 
from src.schemas.context import ExpContext 
from src.configs.lstm import ctx

from pathlib import Path

def run_training(ctx: ExpContext):
    # 设备配置
    device = torch.device(select_gpu())
    logger.info(f"Using device: {device}")

    
    logger.info("Loading Datasets...")
    train_loader, val_loader = malapibert.build_datamodule(ctx)
    
    logger.info(f"Loading Pretrained Model: microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='microsoft/codebert-base', 
        num_labels=ctx.network_config.num_classes
    )
    model.to(device)

    # 优化器与学习率调度器配置
    # BERT 微调通常使用较小的学习率 (2e-5 ~ 5e-5)
    learning_rate = ctx.train_config.learning_rate if ctx.train_config.learning_rate else 2e-5
    epochs = ctx.train_config.epochs
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Warmup 步数通常设为总步数的 10%
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )

    logger.info(f"Start Training: Epochs={epochs}, BatchSize={ctx.train_config.batch_size}, LR={learning_rate}")

    # --- 辅助函数：训练一个 Epoch ---
    def train_epoch(epoch_idx):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch_idx+1}/{epochs}")
        
        for batch in pbar:
            # 将数据移动到 GPU
            # 你的 Dataset 返回的是字典，键名正好对应 BERT 的参数名
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 梯度清零
            model.zero_grad()

            # 前向传播
            # 传入 labels 后，outputs.loss 会自动计算 CrossEntropyLoss
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            # 反向传播
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸，Transformers 训练常用技巧)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        all_preds = []
        all_labels = []
        
        logger.info("Evaluating...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

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
    save_dir = Path(ctx.output_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # 1. 训练
        train_loss, train_acc = train_epoch(epoch)
        logger.info(f"Epoch {epoch+1} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        # 2. 验证
        val_loss, val_acc, val_f1 = eval_epoch(epoch)
        logger.info(f"Epoch {epoch+1} Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # 3. 保存最佳模型 (根据 F1 Score)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = save_dir / "best_model.pt"
            logger.info(f"New Best F1: {best_f1:.4f}. Saving model to {save_path}")
            
            # 保存 model.state_dict() 或者使用 tokenizer.save_pretrained + model.save_pretrained
            torch.save(model.state_dict(), save_path)
            
    logger.info("Training Finished.")
    

if __name__ == "__main__":
    run_training(ctx)
    
    
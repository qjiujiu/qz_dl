from src.utils.logx import logger
from src.light.trainer import Trainer
from src.schemas.context import ExpContext
from src.evaluation.eval_state import EvalState
from src.evaluation.eval_statex import EvalStateX
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Dict


class LatentPGDTrainer(Trainer):
    def __init__(self, ctx: ExpContext, model: nn.Module, train_loader, val_loader=None):
        super().__init__(ctx, model, train_loader, val_loader)
        
        # 读取配置
        self.adv_epsilon = ctx.adv_config.epsilon
        self.adv_alpha = ctx.adv_config.alpha
        self.adv_steps = ctx.adv_config.steps

    def train(self):
        epochs = self.ctx.train_config.epochs
        # 建议：如果是 Free AT 模式，adv_steps 通常不需要设太大（比如 PGD-7），
        # 因为每一步都在更新模型，设太大可能会对这就一个 batch 过拟合。
        logger.info(f"🛡️ Starting Free-AT (Input-PGD) Training: K={self.adv_steps}, Eps={self.adv_epsilon}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0            
            
            with tqdm(self.train_loader, desc=f"FreeEpoch {epoch}/{epochs}") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # ============================================
                    # 1. 准备工作：高斯初始化 Delta
                    # ============================================
                    with torch.no_grad():
                        embeddings_clean = self.model.embedding(x)
                    
                    #  使用高斯噪声初始化，而不是全 0 或均匀分布
                    # std 设为一个较小的值，例如 epsilon 的 1/4 或 0.001
                    delta = torch.zeros_like(embeddings_clean).to(self.device)
                    delta.data.normal_(mean=0, std=self.adv_epsilon / 2) 
                    
                    # 初始化后也要进行一次截断，保证起始点在球内
                    delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                    delta.requires_grad = True

                    # ============================================
                    # 2. Free AT 循环 (同时更新 Delta 和 Model)
                    # ============================================
                    # 以前是：循环K次更新Delta -> 循环外更新1次Model
                    # 现在是：循环K次 -> 每次同时更新 Delta 和 Model
                    for t in range(self.adv_steps):
                        # 2.1 前向传播
                        # 注意：这里我们希望模型能同时学到防御当前 delta 的能力
                        outputs = self.model(x, embed_perturbation=delta)
                        loss = self.loss_fn(outputs, y)
                        
                        # 2.2 反向传播
                        # [关键] 清空模型梯度和 Delta 梯度
                        self.optimizer.zero_grad()
                        if delta.grad is not None:
                            delta.grad.zero_()
                            
                        # 这次 backward 会同时计算 model 参数的梯度 和 delta 的梯度
                        loss.backward()
                        
                        # ============================================
                        # 2.3 更新扰动 Delta (梯度上升 Gradient Ascent)
                        # ============================================
                        if delta.grad is not None:
                            g = delta.grad.detach()
                            # 归一化 (FGM style)
                            norm = torch.norm(g, dim=-1, keepdim=True)
                            norm = torch.clamp(norm, min=1e-8)
                            d = g / norm
                            
                            # 更新 delta
                            delta.data = delta.data + self.adv_alpha * d
                            delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                        
                        # ============================================
                        # 2.4 更新模型参数 (梯度下降 Gradient Descent)
                        # ============================================
                        # [修改点 2] 直接利用刚才 backward 算出来的模型梯度更新参数
                        # 相当于在一个 batch 上进行了 adv_steps 次微调
                        self.optimizer.step()
                        
                        # 记录最后一步的 Loss 用于展示
                        current_loss = loss.item()

                    # 更新统计数据 (取最后一次迭代的 loss)
                    total_loss += current_loss
                    pbar.set_postfix(loss=current_loss)

            # 评估与保存逻辑保持不变...
            avg_loss = total_loss / len(self.train_loader)
            self._run_epoch_evaluation(epoch, avg_loss, epochs)
            if self.scheduler:
                self.scheduler.step()
            self._save_trace()

        self._save_checkpoint(suffix="last")
        logger.info("Free-AT Training Finished.")
    
    def _run_epoch_evaluation(self, epoch, avg_loss, epochs):
        # 直接调用 evaluate (内部已强制为对抗评估)
        metrics = self.evaluate()
        
        # 记录 History
        self.history['train_loss'].append(avg_loss)
        self.history['val_metrics'].append(metrics)
        
        logger.info(f"Epoch {epoch}/{epochs} | Adv Loss: {avg_loss:.4f}")
        logger.info(f"  Robust Metrics: {metrics}")
        
    def evaluate(self) -> Dict:
        """ 评估：对验证集进行同样的 Input-PGD 攻击后测试准确率 """
        self.model.eval()
        all_preds, all_labels = [], []
        
        # 验证集攻击通常步数可以少一点，或者保持一致
        eval_steps = self.adv_steps 
        
        for x, y in tqdm(self.val_loader, desc="Robust Eval"):
            x, y = x.to(self.device), y.to(self.device)
            
            # --- 生成攻击样本 ---
            # 必须开启梯度来生成 Delta
            with torch.enable_grad():
                # 1. 初始化 Delta
                with torch.no_grad():
                    embeddings_clean = self.model.embedding(x)
                delta = torch.zeros_like(embeddings_clean).to(self.device)
                delta.requires_grad = True
                
                # 2. 攻击循环
                # 注意：为了兼容 CuDNN RNN，这里需不需要 model.train()？
                # 理论上只需要 embedding 层之后的梯度。
                # 保险起见，生成 delta 期间切到 train，最后 inference 切回 eval
                self.model.train() 
                
                for _ in range(eval_steps):
                    outputs = self.model(x, embed_perturbation=delta)
                    loss = self.loss_fn(outputs, y)
                    loss.backward()
                    
                    if delta.grad is not None:
                        g = delta.grad.detach()
                        # 简单的 PGD 更新
                        delta.data = delta.data + self.adv_alpha * torch.sign(g)
                        delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                        delta.grad.zero_()
            
            # --- 正式推断 ---
            self.model.eval() # 切回 eval 模式 (关闭 Dropout)
            delta = delta.detach()
            
            with torch.no_grad():
                # 传入生成的 delta 进行预测
                outputs = self.model(x, embed_perturbation=delta)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        # 计算指标
        state = EvalState(
            labels=np.concatenate(all_labels),
            predicts=np.concatenate(all_preds)
        )
        
        metrics = state.calculate()
            
        return metrics
    

class LatentPGDMultilabelTrainer(Trainer):
    def __init__(self, ctx: ExpContext, model: nn.Module, train_loader, val_loader=None):
        super().__init__(ctx, model, train_loader, val_loader)
        
        # 读取对抗配置
        self.adv_epsilon = ctx.adv_config.epsilon
        self.adv_alpha = ctx.adv_config.alpha
        self.adv_steps = ctx.adv_config.steps
        
        # 确保 Loss 是 BCEWithLogitsLoss (也可以在 build_loss_fn 里做，这里双重保险)
        if not isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            logger.warning("LatentPGDMultilabelTrainer should use BCEWithLogitsLoss! Forcing switch.")
            self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self):
        epochs = self.ctx.train_config.epochs
        logger.info(f"🛡️ [Multi-label] Starting Free-AT Training: K={self.adv_steps}, Eps={self.adv_epsilon}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0            
            
            with tqdm(self.train_loader, desc=f"FreeEpoch {epoch}/{epochs}") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # [多标签修正] BCE Loss 要求 target 为 float
                    y_float = y.float() 

                    # ============================================
                    # 1. 初始化 Delta (高斯噪声)
                    # ============================================
                    with torch.no_grad():
                        embeddings_clean = self.model.embedding(x)
                    
                    # 高斯初始化
                    delta = torch.zeros_like(embeddings_clean).to(self.device)
                    delta.data.normal_(mean=0, std=self.adv_epsilon / 2) 
                    delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                    delta.requires_grad = True

                    # ============================================
                    # 2. Free AT 循环
                    # ============================================
                    for t in range(self.adv_steps):
                        # 2.1 前向传播
                        outputs = self.model(x, embed_perturbation=delta)
                        
                        # [多标签修正] 使用 float 标签计算 Loss
                        loss = self.loss_fn(outputs, y_float)
                        
                        # 2.2 反向传播
                        self.optimizer.zero_grad()
                        if delta.grad is not None:
                            delta.grad.zero_()
                            
                        loss.backward()
                        
                        # ============================================
                        # 2.3 更新扰动 Delta (梯度上升)
                        # ============================================
                        if delta.grad is not None:
                            g = delta.grad.detach()
                            norm = torch.norm(g, dim=-1, keepdim=True)
                            norm = torch.clamp(norm, min=1e-8)
                            d = g / norm
                            
                            delta.data = delta.data + self.adv_alpha * d
                            delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                        
                        # ============================================
                        # 2.4 更新模型参数 (梯度下降)
                        # ============================================
                        self.optimizer.step()
                        
                        current_loss = loss.item()

                    total_loss += current_loss
                    pbar.set_postfix(loss=current_loss)

            # 评估与保存
            avg_loss = total_loss / len(self.train_loader)
            self._run_epoch_evaluation(epoch, avg_loss, epochs)
            if self.scheduler:
                self.scheduler.step()
            self._save_trace()

        self._save_checkpoint(suffix="last-adv")
        logger.info("Free-AT Multi-label Training Finished.")
    
    def _run_epoch_evaluation(self, epoch, avg_loss, epochs):
        # 调用重写后的多标签评估
        metrics = self.evaluate()
        
        self.history['train_loss'].append(avg_loss)
        self.history['val_metrics'].append(metrics)
        
        logger.info(f"Epoch {epoch}/{epochs} | Adv Loss: {avg_loss:.4f}")
        # 这里 metrics 包含 micro_f1, macro_f1 等
        logger.info(f"  Robust Metrics: {metrics}")
        
    def evaluate(self) -> Dict:
        """ 
        [多标签专用评估]
        对验证集进行 PGD 攻击，然后计算多标签指标 (Micro-F1 等)
        """
        self.model.eval()
        all_preds, all_labels = [], []
        
        eval_steps = self.adv_steps 
        
        for x, y in tqdm(self.val_loader, desc="Robust Eval (Multi-label)"):
            x, y = x.to(self.device), y.to(self.device)
            y_float = y.float() # BCE 需要 float
            
            # --- 1. 生成攻击样本 (PGD) ---
            with torch.enable_grad():
                with torch.no_grad():
                    embeddings_clean = self.model.embedding(x)
                    
                delta = torch.zeros_like(embeddings_clean).to(self.device)
                delta.requires_grad = True
                
                # 切换到 train 模式以确保梯度回传 (针对特定 RNN/Dropout 层)
                self.model.train() 
                
                for _ in range(eval_steps):
                    outputs = self.model(x, embed_perturbation=delta)
                    loss = self.loss_fn(outputs, y_float)
                    loss.backward()
                    
                    if delta.grad is not None:
                        g = delta.grad.detach()
                        delta.data = delta.data + self.adv_alpha * torch.sign(g)
                        delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                        delta.grad.zero_()
            
            # --- 2. 正式推断 ---
            self.model.eval() # 必须切回 eval
            delta = delta.detach()
            
            with torch.no_grad():
                outputs = self.model(x, embed_perturbation=delta)
            
            # 获取 Logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # ============================================
            # [多标签核心修改] Sigmoid + Threshold
            # ============================================
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()   # 转为 0/1 整数向量
            
            all_preds.append(preds.cpu().numpy())
            # y 已经在 dataset 里是 multi-hot float 了，转 int 以便 sklearn 计算
            all_labels.append(y.cpu().int().numpy())

        # --- 3. 计算指标 ---
        # 使用 EvalStateX (支持多标签)
        state = EvalStateX(
            labels=np.concatenate(all_labels),
            predicts=np.concatenate(all_preds)
        )
        
        return state.calculate()
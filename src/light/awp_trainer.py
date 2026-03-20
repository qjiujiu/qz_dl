
from src.utils.logx import logger
from src.schemas.context import ExpContext, AdvConfig
from src.schemas.block_enums import AttackType
from src.evaluation.eval_state import EvalState
from src.evaluation.eval_statex import EvalStateX

from src.light.trainer import Trainer 

from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np


def build_attacker(cfg: AdvConfig, model: nn.Module):
    from src.utils.weight_perbutation import FGM 
    from src.utils.weight_perbutation import PGD    
    
    attacker_creator = {
        AttackType.FGM: lambda: FGM(model, epsilon=cfg.epsilon),
        AttackType.PGD: lambda: PGD(model, epsilon=cfg.epsilon, alpha=cfg.alpha),  
    }
    creator = attacker_creator.get(cfg.adv_type)
    
    if not creator:
        raise ValueError(f"Unknown attack type: {cfg.adv_type}")
    
    return creator()


class AWPTrainer(Trainer):
    def __init__(self, ctx: ExpContext, model: nn.Module, train_loader, val_loader=None):
        super().__init__(ctx, model, train_loader, val_loader)
        
        # 1. 初始化攻击器
        self.attacker = build_attacker(ctx.adv_config, self.model)
        
        # 2. 读取对抗参数
        self.adv_epsilon = getattr(self.ctx.adv_config, 'epsilon', 1.0)
        self.adv_emb_name = 'embedding'
        
        self.adv_steps = ctx.adv_config.steps
        self.adv_alpha = ctx.adv_config.alpha

        # 如果是 FGM，强制 K=1 (因为其并非迭代攻击)
        if self.ctx.adv_config.adv_type == AttackType.FGM:
            self.adv_steps = 1

    def train(self):
        epochs = self.ctx.train_config.epochs
        logger.info(f"🛡️ Starting ADVERSARIAL training: {self.task_id} (Type: {self.ctx.adv_config.adv_type}, K={self.adv_steps})")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_adv_loss = 0            
            
            with tqdm(self.train_loader, desc=f"AdvTrain {epoch}/{epochs}") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)

                    # 1. 正常前向传播，获取初始梯度
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y)
                    loss.backward()
                    
                    # =========================================
                    # K 次攻击迭代循环 (PGD 核心)
                    # =========================================
                    for t in range(self.adv_steps):
                        # 标记是否为第一步 (用于备份参数)
                        is_first = (t == 0)
                        
                        # 2. 实施攻击 (Attack)
                        self.attacker.attack(
                            epsilon=self.adv_epsilon, 
                            alpha=self.adv_alpha, 
                            emb_name=self.adv_emb_name,
                            is_first_attack=is_first
                        )
                        
                        # 3. 如果不是最后一步，需要重新计算梯度用于下一步攻击
                        # (因为参数变了，梯度也要跟着变，这就是 PGD 慢的原因)
                        if t != self.adv_steps - 1:
                            self.model.zero_grad()
                            outputs_tmp = self.model(x)
                            loss_tmp = self.loss_fn(outputs_tmp, y)
                            loss_tmp.backward()
                    
                    # =========================================
                    # 最终对抗训练更新
                    # =========================================
                    # 4. 对抗样本前向传播
                    outputs_adv = self.model(x)
                    loss_adv = self.loss_fn(outputs_adv, y)
                    
                    # 5. 反向传播 (此时使用的是基于"最坏扰动"计算出的梯度)
                    # 注意：如果您希望只用对抗梯度更新，就先 zero_grad；
                    # 如果希望 原始梯度 + 对抗梯度 混合更新，就不 zero_grad。
                    # 这里采用纯对抗训练 (Madry风格)，先清空之前的梯度
                    # self.model.zero_grad() 
                    loss_adv.backward()
                    
                    # 6. 恢复原始参数 (必须恢复，不然下次就是在脏数据上继续走了)
                    self.attacker.restore(emb_name=self.adv_emb_name)
                    
                    # 7. 参数更新
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_adv_loss += loss_adv.item()
                    pbar.set_postfix(adv_loss=loss_adv.item())
            
            # --- 评估 ---
            avg_loss = total_adv_loss / len(self.train_loader)
            self._run_epoch_evaluation(epoch, avg_loss, epochs)
            
            if self.scheduler:
                self.scheduler.step()
                
            self._save_trace()

        self._save_checkpoint(suffix="last")
        logger.info("Adversarial Training Finished.")
        

    def _run_epoch_evaluation(self, epoch, avg_loss, epochs):
        # 直接调用 evaluate (内部已强制为对抗评估)
        metrics = self.evaluate()
        
        # 记录 History
        self.history['train_loss'].append(avg_loss)
        self.history['val_metrics'].append(metrics)
        
        logger.info(f"Epoch {epoch}/{epochs} | Adv Loss: {avg_loss:.4f}")
        logger.info(f"  Robust Metrics: {metrics}")

    def evaluate(self) -> Dict:
        """ 
        纯对抗评估函数: 
        不关心干净样本，强制对验证集所有数据进行攻击后再评估。
        """
        all_preds, all_labels = [], []
        
        # 必须开启梯度 (torch.enable_grad) 才能生成对抗样本
        # 
        with torch.enable_grad():
            for x, y in tqdm(self.val_loader, desc="Robust Eval"):
                x, y = x.to(self.device), y.to(self.device)
                
                # ===========================
                # 1. 切换 Train 模式 (为了 CuDNN RNN 兼容性)
                # ===========================
                self.model.train()
                self.model.zero_grad()
                
                # ===========================
                # 2. 生成攻击 (Generate Attack)
                # ===========================
                # 前向 + 反向获取梯度
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                
                # 施加扰动
                self.attacker.attack(epsilon=self.adv_epsilon, emb_name=self.adv_emb_name)
                
                # ===========================
                # 3. 对抗推断 (Inference on Adv)
                # ===========================
                # 切换回 Eval 模式进行正式预测 (关闭 Dropout 等)
                self.model.eval()
                
                with torch.no_grad():
                    outputs_adv = self.model(x)
                
                # ===========================
                # 4. 恢复现场 (Restore)
                # ===========================
                self.attacker.restore(emb_name=self.adv_emb_name)
                self.model.zero_grad() 
                
                # 收集结果
                logits = outputs_adv.logits if hasattr(outputs_adv, 'logits') else outputs_adv
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
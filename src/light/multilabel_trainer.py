from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.evaluation.eval_statex import EvalStateX
from src.light.trainer import Trainer 

from typing import Dict

import torch
import numpy as np



class MultilabelTrainer(Trainer):
    """
    专门适配【多标签多分类】任务的训练器。
    继承自标准 Trainer，仅重写评估逻辑。
    """
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # 遍历验证集
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            #  前向传播 (输出 Logits)
            logits = self.model(x)
            
            # Logits -> Probabilities (0~1)
            probs = torch.sigmoid(logits)
            
            # Thresholding (> 0.5) -> 0/1 Indicators, 使用 .int() 或 .float() 均可，sklearn 均可处理
            preds = (probs > 0.5).int()
            
            # 收集结果 (转为 numpy)
            all_preds.append(preds.cpu().numpy())
            
            # 标签向量 y 已是 multi-hot 向量了，直接转 numpy 即可, 转为 int, 确保类型一致于 preds 
            all_labels.append(y.cpu().int().numpy())

        #  拼接所有 Batch
        y_true_all = np.concatenate(all_labels)  # Shape: [Total_Val_Samples, Num_Classes]
        y_pred_all = np.concatenate(all_preds)   # Shape: [Total_Val_Samples, Num_Classes]
        
        #  使用 EvalStateX 计算指标
        state = EvalStateX(
            labels=y_true_all,
            predicts=y_pred_all
        )
        
        return state.calculate()
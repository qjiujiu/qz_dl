from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from typing import List, Dict, Optional, Union
import numpy as np


class EvalStateX(BaseModel):
    """ 
    统一评估状态对象 (支持单标签 & 多标签)
    核心逻辑：利用 sklearn 自动处理不同维度的输入
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
   
    # 标签与预测
    # 单标签: shape=[N], int64
    # 多标签: shape=[N, C], int/float (0/1 indicators)
    labels: np.ndarray
    predicts: np.ndarray
    
    # 辅助元信息
    total_classes: Optional[int] = Field(default=None, description="可选：类别数")

    @field_serializer('labels', 'predicts')
    def serialize_ndarray(self, v: np.ndarray) -> List:
        return v.tolist() 

    def _fmt(self, val: float) -> float:
        return round(val * 100, 2)

    def calculate(self) -> Dict:
        """ 智能计算指标 """
        y_true = self.labels
        y_pred = self.predicts
        
        # 自动判断任务类型
        # 如果是多维数组且第二维 > 1，或者是一个 0/1 二维矩阵，则是多标签
        is_multilabel = (y_true.ndim == 2) and (y_true.shape[1] > 1)
        
        # 准备 sklearn 参数
        # zero_division=0 防止除以零报错
        # average=None 用于获取每个类别的独立指标 (用于调试或扩展)
        common_kwargs = {"zero_division": 0}

        # 计算 Accuracy
        # - 单标签: 预测对即得分
        # - 多标签: Subset Accuracy (极其严苛，必须全对才得分)
        acc = accuracy_score(y_true, y_pred)

        # 计算 Micro (全局) 指标
        # Micro F1 在多标签中非常常用
        micro_p = precision_score(y_true, y_pred, average='micro', **common_kwargs)
        micro_r = recall_score(y_true, y_pred, average='micro', **common_kwargs)
        micro_f1 = f1_score(y_true, y_pred, average='micro', **common_kwargs)

        # 计算 Macro (宏平均) 指标
        # 对不平衡数据非常重要
        macro_p = precision_score(y_true, y_pred, average='macro', **common_kwargs)
        macro_r = recall_score(y_true, y_pred, average='macro', **common_kwargs)
        macro_f1 = f1_score(y_true, y_pred, average='macro', **common_kwargs)

            
        _fmt = self._fmt
        
        metrics = {
            "task_type": "multilabel" if is_multilabel else "multiclass",
            "acc": _fmt(acc),
            "micro": {    
                "p": _fmt(micro_p),
                "r": _fmt(micro_r),
                "f1": _fmt(micro_f1),
            }, 
            "macro": {
                "p": _fmt(macro_p),
                "r": _fmt(macro_r),
                "f1": _fmt(macro_f1)
            }
        }
        
        # 若是多标签多分类, 计算每个样本的 F1 然后平均，反映了“每个样本被预测正确的程度”
        if is_multilabel:
            samples_f1 = f1_score(y_true, y_pred, average='samples', **common_kwargs)
            metrics["samples_f1"] = _fmt(samples_f1)
            
        return metrics
from src.eval.state import State

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import numpy as np



class EvalState(BaseModel):
    """ labels/predicts 必须传入 numpy.ndarray:
        - shape=[N]：类别 id
        - shape=[N,C]：one-hot/prob/logits
    """
    
    # 允许 numpy 类型的字段
    model_config = ConfigDict(arbitrary_types_allowed=True)
   
    labels: np.ndarray
    predicts: np.ndarray
    total_classes: Optional[int] = Field(default=None, description="可选：类别数，若不提供则自动推断")

    # ===== 内部工具：统一转为类别 ID =====
    @staticmethod
    def _to_ids(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.astype(np.int64)
        if x.ndim == 2:
            return np.argmax(x, axis=1).astype(np.int64)
        raise ValueError(f"expect 1D or 2D array, got shape={x.shape}")

    @property
    def y_true(self) -> np.ndarray:
        return self._to_ids(self.labels)

    @property
    def y_pred(self) -> np.ndarray:
        return self._to_ids(self.predicts)

    @property
    def num_labels(self) -> int:
        if self.total_classes is not None:
            return int(self.total_classes)
        # 尝试推断
        if self.labels.ndim == 2: return int(self.labels.shape[1])
        if self.predicts.ndim == 2: return int(self.predicts.shape[1])
        # 基于最大 ID 推断
        return int(max(self.y_true.max(initial=-1), self.y_pred.max(initial=-1)) + 1)

    @property
    def confusion_matrix(self) -> np.ndarray:
        """ 计算混淆矩阵，行是 True，列是 Pred """
        n = self.num_labels
        cm = np.zeros((n, n), dtype=np.int64)
        np.add.at(cm, (self.y_true, self.y_pred), 1)
        return cm

    # ===== 核心：基于混淆矩阵的高效统计 =====
    @property
    def _cm_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 一次性计算所有类别的 TP, FP, FN, TN, 
            返回值是一个数组元素 (TPs, FPs, TNs, FNs), 每个都是 shape=[num_labels] 的数组
        """
        cm = self.confusion_matrix
        total_samples = cm.sum()
        
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP  # 列求和 - TP
        FN = cm.sum(axis=1) - TP  # 行求和 - TP
        TN = total_samples - (TP + FP + FN)
        
        return TP, FP, TN, FN

    def _state_from_stats(self, idx: int, stats: tuple) -> State:
        """ 从缓存的 stats 中提取特定类别的 State """
        TPs, FPs, TNs, FNs = stats
        return State(
            TP=int(TPs[idx]),
            FP=int(FPs[idx]),
            TN=int(TNs[idx]),
            FN=int(FNs[idx])
        )

    # ===== Micro Average (微平均) =====
    # 微平均是将所有类别的 TP/FP/FN/TN 累加，构成一个全局的 State
    @property
    def micro_state(self) -> State:
        TPs, FPs, TNs, FNs = self._cm_stats
        return State(
            TP=int(TPs.sum()),
            FP=int(FPs.sum()),
            TN=int(TNs.sum()),
            FN=int(FNs.sum())
        )

    @property
    def micro_precision(self) -> float:
        return self.micro_state.precision

    @property
    def micro_recall(self) -> float:
        return self.micro_state.recall

    @property
    def micro_f1_score(self) -> float:
        return self.micro_state.f1_score
    
    # ===== Macro Average (宏平均) =====
    # 宏平均是先计算每个类别的指标，然后求算术平均
    @property
    def macro_precision(self) -> float:
        stats = self._cm_stats
        n = self.num_labels
        if n == 0:
            return 0.0
        
        # 列表推导式计算每个类别的 precision 然后求平均
        return float(np.mean([self._state_from_stats(k, stats).precision for k in range(n)]))

    @property
    def macro_recall(self) -> float:
        stats = self._cm_stats
        n = self.num_labels
        if n == 0: 
            return 0.0
        return float(np.mean([self._state_from_stats(k, stats).recall for k in range(n)]))

    @property
    def macro_f1_score(self) -> float:
        stats = self._cm_stats
        n = self.num_labels
        if n == 0: 
            return 0.0
        return float(np.mean([self._state_from_stats(k, stats).f1_score for k in range(n)]))

    # ===== 每个类别的 State =====
    def state_one_vs_rest(self, k: int) -> State:
        return self._state_from_stats(k, self._cm_stats)
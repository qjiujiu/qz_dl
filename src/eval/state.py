from __future__ import annotations

from typing import ClassVar, Optional, Any, List
from pydantic import BaseModel, ConfigDict, Field



class State(BaseModel):
    _eps: ClassVar[float] = 1e-10  # 防止除零错误

    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    @property
    def precision(self) -> float:
        return float(self.TP / (self.TP + self.FP + self._eps))

    @property
    def recall(self) -> float:
        return float(self.TP / (self.TP + self.FN + self._eps))

    @property
    def f1(self) -> float:
        p, r =self.precision, self.recall
        return float(2 * p * r / (p + r + self._eps))

    @property
    def accuracy(self) -> float:
        total = self.TP + self.TN + self.FP + self.FN
        return float((self.TP + self.TN) / (total + self._eps))
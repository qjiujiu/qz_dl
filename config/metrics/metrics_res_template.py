from dataclasses import dataclass, asdict



@dataclass
class ClassificationResult:
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    verbose: bool = False
    
    def __post_init__(self):
        pass


    def as_dict(self):
        return asdict(self)

    def __str__(self):
        parts = []
        
        if self.accuracy is not None:
            parts.append(f"Acc: {self.accuracy * 100:.2f}%")
        if self.precision is not None:
            parts.append(f"Prec: {self.precision * 100:.2f}%")
        if self.recall is not None:
            parts.append(f"Recall: {self.recall * 100:.2f}%")
        if self.f1 is not None:
            parts.append(f"F1: {self.f1 * 100:.2f}%")
        
        # 如果是一个空对象则什么都不打印，返回结果将是一个空字符串
        return ", ".join(parts)
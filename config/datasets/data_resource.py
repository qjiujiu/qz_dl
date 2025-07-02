import numpy as np

from dataclasses import dataclass, field, asdict
from typing import Optional

from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataResource:
    train_dataset: Dataset
    test_dataset: Dataset
    batch_size: int
    X_train: list = field(default_factory=list)
    X_test: list = field(default_factory=list)
    y_train: list = field(default_factory=list)
    y_test: list = field(default_factory=list)
    vocab: dict = None 

    def __str__(self):
        # 打印train_dataset和test_dataset的shape
        return f"train_dataset size: {len(self.train_dataset)}, test_dataset size: {len(self.test_dataset)}"

    def __post_init__(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    # 转为 sklean 支持的格式
    def to_sklearn_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
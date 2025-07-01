
from dataclasses import dataclass, asdict
from typing import Optional

from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class DataResource:
    train_loader: DataLoader
    test_loader: DataLoader
    vocab: dict = None 

    # TODO 重写这个对象
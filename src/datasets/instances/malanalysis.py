from src.utils.logx import logger
from src.schemas.context import ExpContext, DataConfig
from src.datasets.instances.malapi2019 import (
    _print_data_distribution
)
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Union
from pathlib import Path

import torch
import pandas as pd


class MalAnalysisDataset(Dataset):
    def __init__(self, csv_path: Union[str, Path]):
        """ 
        二分类: 恶意软件分析数据集
        数据集: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        # 读取数据, CSV 结构: hash, t_0, ..., t_99, malware
        df = pd.read_csv(self.csv_path)

        # 提取特征 (t_0 到 t_99)
        # 过滤掉 'hash'/'malware' 列，剩下的即可时间步特征
        feature_cols = [c for c in df.columns if c not in ['hash', 'malware']]
        
        # NOTE: 转换为 numpy 数组再转 tensor, 
        # NOTE: 其中, API ID 是离散整数，适用于的 Embedding layer，必须使用 long 类型
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.long)
        
        # 提取标签 (malware)
        self.labels = torch.tensor(df['malware'].values, dtype=torch.long)

        # 记录一些元信息
        self.num_samples = len(df)
        self.seq_len = len(feature_cols)
        
        # 记录词表
        self.max_api_id = int(self.features.max().item())
        self.min_api_id = int(self.features.min().item())
        self.vocab_size = self.max_api_id + 1
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """ 返回: (sequence_indices, label)
        - Seq: shape [100]
        - label: scalar (0 or 1)
        """
        return self.features[idx], self.labels[idx]
    



def build_datamodule(ctx: ExpContext) -> Tuple[DataLoader, DataLoader]:
    cfg: DataConfig = ctx.data_config
    
    # 拼接完整的文件路径
    csv_filename ="dynamic_api_call_seq100.csv"
    data_path = ctx.data_config.data_dir / csv_filename

    # 实例化完整数据集
    dataset = MalAnalysisDataset(csv_path=data_path)

    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * cfg.test_size)
    train_size = total_size - val_size
    
    
    # 使用配置中的 seed 确保划分可复现
    generator = torch.Generator().manual_seed(ctx.train_config.seed)
    
    train_subset, val_subset = random_split(
        dataset=dataset, 
        lengths=[train_size, val_size],
        generator=generator
    )
    
    
    # NOTE 此处打印一下恶意软件、良性软件的比例
    _print_data_distribution(labels=dataset.labels, title="Total")
    
    train_labels = dataset.labels[train_subset.indices]
    _print_data_distribution(train_labels, title="Train Set")

    val_labels = dataset.labels[val_subset.indices]
    _print_data_distribution(val_labels, title="Validation Set")
    

    logger.info(f"API ID Range :[{dataset.min_api_id}, {dataset.max_api_id}], vocab size: {dataset.vocab_size}")


    # 构建 DataLoader, 通过 ctx 中读取 num_workers, pin_memory, batch_size
    loader_args = dict(
        batch_size=ctx.train_config.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    
    return train_loader, val_loader

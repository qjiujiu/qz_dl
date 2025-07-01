import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

def load_clean_embedding_dataset(cache_path, batch_size, test_ratio=0.2):
    """
    加载干净嵌入表示数据集，并划分训练集与测试集（但主要用于测试阶段）。
    返回：train_loader, test_loader（你可在需要时仅使用 test_loader）
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"未找到缓存文件：{cache_path}")

    # 加载嵌入特征和标签
    clean_emb, clean_labels = torch.load(cache_path)
    clean_dataset = TensorDataset(clean_emb, clean_labels)

    # 划分训练集和测试集
    test_size = int(test_ratio * len(clean_dataset))
    train_size = len(clean_dataset) - test_size
    train_dataset, test_dataset = random_split(
        clean_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # 封装为 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
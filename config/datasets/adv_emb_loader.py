import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

def load_adversarial_dataset(emb_path, batch_size, train_ratio=0.8):
    """
    从给定的对抗样本缓存中加载数据并划分训练与测试集
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"缓存文件 {emb_path} 不存在")

    emb_tensor, label_tensor = torch.load(emb_path)
    dataset = TensorDataset(emb_tensor, label_tensor)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        # 加载对抗样本数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
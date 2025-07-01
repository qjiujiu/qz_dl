import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config.logger import logger
from config.datasets.text_preprocessing import (
    build_vocab,
    text_to_indices, 
    pad_sequence
)


logger.is_debug(True)


# 加载数据集，并将文本转为索引的形式
class MalAPITextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_indices = text_to_indices(text, self.vocab)
        padded_text = pad_sequence(text_indices, self.max_len)
        return torch.tensor(padded_text), torch.tensor(label)





def load(text_file, labels_file, split = True, cache_dir='data/malapi2019/preprocessed', test_size=0.2, random_state=42):
    # 类别标签到整数的映射
    label_map = {
        'Spyware': 0,
        'Downloader': 1,
        'Trojan': 2,
        'Worms': 3,
        'Adware': 4,
        'Dropper': 5,
        'Virus': 6,
        'Backdoor': 7
    }


    print("📦 正在首次处理并缓存数据集...")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [label.strip() for label in f.readlines()]


    # 将标签从字符串转换为整数
    labels = [label_map[label] for label in labels]

    # 将数据分割成训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state)

    # 缓存词表
    vocab = build_vocab(train_texts)

    
    logger.debug(f"词表大小：{len(vocab)}")
    logger.debug(f"训练集文本标签对: {len(train_texts)}, 测试集文本标签对: {len(test_texts)}")

    # 创建训练和测试数据集
    # 传递的是划分好的文本和标签，而不是文件路径
    train_dataset = MalAPITextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
    test_dataset = MalAPITextDataset(texts=test_texts, labels=test_labels, vocab=vocab)
   

    return train_dataset, test_dataset, vocab

from src.utils.logx import logger
from src.utils.dumps import dump_to_json, load_from_json
from src.utils.text_preprocessing import dedup_preprocess
from src.schemas.context import ExpContext, DataConfig
from src.utils.text_preprocessing import build_vocab
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict, Callable

import json
import torch
import numpy as np


# 多标签-多分类任务(十五分类)
CLASS_NAMES = [
    "benign", "malware", "trojan", "banker", "pua", 
    "downloader", "adware", "dropper", "spyware", "virus", 
    "miner", "ransomware", "worm", "hacktool", "fakeav"
]

LABEL2ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}



class MalDynamic(Dataset):
    def __init__(self, 
            api_sequences: List[List[str]], 
            labels: List[List[int]], 
            vocab: Dict[str, int], 
            max_len: int = 200
        ):
        """ 数据来源
            api_sequences: API 调用序列列表，每个元素是 List[str]
            labels: 标签列表，每个元素是 0/1 的 List[int] (长度15)
            vocab: 词表
            max_len: 序列最大长度
        """
        self.api_sequences = api_sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
        # 缓存特殊索引
        self.unk_idx = self.vocab.get('<unk>', 0)
        self.pad_idx = self.vocab.get('<pad>', 1)

    def __len__(self):
        return len(self.api_sequences)

    def __getitem__(self, idx):
        # 获取原始数据
        tokens = dedup_preprocess(self.api_sequences[idx]) 
        label_vec = self.labels[idx]
        
        # (Token -> ID)
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]
        
        # 截断与填充
        if len(indices) >= self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.pad_idx] * (self.max_len - len(indices))
            
        # X: API 序列，用于 Embedding
        x_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Y: 多标签向量，用于 BCEWithLogitsLoss
        y_tensor = torch.tensor(label_vec, dtype=torch.float)
        
        return x_tensor, y_tensor



def _print_multilabel_distribution(labels: np.ndarray, title: str = "Dataset"):
    """ 专门用于打印多标签数据的分布
        labels shape: [N_samples, N_classes]
    """
    total_samples = labels.shape[0]
    class_counts = labels.sum(axis=0) 
    
    # 按列求和，得到每个类别的正样本数量
    logger.info(f"--- {title} Distribution (Total Samples: {total_samples}) ---") 
    for idx, count in enumerate(class_counts):
        class_name = ID2LABEL[idx]
        if count > 0: # 只打印出现过的类别
            percentage = (count / total_samples) * 100
            logger.info(f"{class_name:<12}: {int(count):5d} pos samples, {percentage:6.2f}%")
            


def build_datamodule(ctx: ExpContext) -> Tuple[DataLoader, DataLoader]:
    """  构建动态分析数据加载器
    1. 读取 benign.json, 368.json, 389.json
    2. 合并数据 & 构建词表
    3. 划分数据集 & 封装 DataLoader
    """
    data_dir = ctx.data_config.data_dir 
    
    # 定义要读取的文件
    target_files = ['benign.json', '368.json', '389.json']
    
    all_apis = []
    all_labels = []
    
    # 读取并合并数据
    logger.info("Loading JSON files...")
    for file_name in target_files:
        file_path = data_dir / "Processed" / file_name
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_apis.extend(data["apis"])
                all_labels.extend(data["labels"])
                logger.info(f"Loaded {len(data['apis'])} samples from {file_name}")
        except Exception as e:
            logger.error(f"Error reading {file_name}: {e}")

    if not all_apis:
        raise ValueError("No data loaded! Please check data paths.")

    # 构建词表, 存储词表大小
    vocab_path = data_dir / "vocab.json"
    
    if vocab_path.exists():
        logger.info(f"Loading existing vocabulary from {vocab_path}")
        vocab = load_from_json(vocab_path)
    else:
        logger.info(f"Vocab not found at {vocab_path}, building new one...")
        vocab = build_vocab(all_apis, min_freq=ctx.data_config.nlp_config.min_freq)
        dump_to_json(vocab, vocab_path)
            
    
    # 获取词表大小
    ctx.network_config.vocab_size = len(vocab)
    logger.info(f"Vocab Size: {len(vocab)}")

    # 实例化 Dataset
    dataset = MalDynamic(
        api_sequences=all_apis,
        labels=all_labels,
        vocab=vocab,
        max_len=ctx.data_config.max_len
    )

    # 划分训练/验证集
    total_size = len(dataset)
    val_size = int(total_size * ctx.data_config.test_size)
    train_size = total_size - val_size
    
    
    generator = torch.Generator().manual_seed(ctx.train_config.seed)
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    
    # 因为 all_labels 本身就是 list，可以直接转 numpy
    np_labels = np.array(all_labels)
    _print_multilabel_distribution(np_labels, title="Total Dataset")
    
    # 打印验证集的分布 (通过索引切片)
    val_indices = val_subset.indices
    _print_multilabel_distribution(np_labels[val_indices], title="Validation Set")

    # 构建 DataLoader
    loader_args = dict(
        batch_size=ctx.train_config.batch_size,
        num_workers=ctx.data_config.num_workers,
        pin_memory=ctx.data_config.pin_memory,
    )
    
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)

    return train_loader, val_loader

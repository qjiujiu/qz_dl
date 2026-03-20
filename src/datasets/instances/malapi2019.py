from src.utils.logx import logger
from src.utils.dumps import read_pickle, write_pickle
from src.utils.text_preprocessing import (
    build_vocab, 
    default_preprocess, 
    dedup_preprocess,
)
from src.schemas.context import ExpContext, DataConfig

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Callable
from collections import Counter
from pathlib import Path
import torch
import numpy as np



# 正向映射
LABEL_MAP = {
    'Spyware': 0, 'Downloader': 1, 'Trojan': 2, 'Worms': 3,  
    'Adware': 4, 'Dropper': 5, 'Virus': 6, 'Backdoor': 7
}

# 反向映射, 主要用于打印日志
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

class MalAPITextDataset(Dataset):
    """ 
    https://www.kaggle.com/datasets/focatak/malapi2019
    """
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 200, text_pipeline: Callable[[str], List[str]] = default_preprocess):
        self.texts: List[int] = texts
        self.labels: List[int] = labels
        self.vocab: Dict = vocab
        self.max_len: int = max_len
        self.text_pipline: Callable[[str], List[str]] = text_pipeline
        
        # 缓存一下特殊的索引，避免由 dict 查找带来的开销
        self.unk_idx = self.vocab.get('<unk>', 0)
        self.pad_idx = self.vocab.get('<pad>', 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词 (逻辑与词表构建保持一致 build_vocab)
        tokens = self.text_pipline(text)
        
        # 转索引 (查不到的给 unk_idx)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # 截断与填充, 填充使用 pad_idx (1)
        if len(indices) >= self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.pad_idx] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)



def _load_raw_text_data(data_dir: Path, test_size: float = 0.2) -> Tuple:
    """
    只负责：读取原始 txt -> 标签映射 -> 切分数据集
    """
    text_path = data_dir / "all_analysis_data.txt"
    label_path = data_dir / "labels.txt"
    
    if not text_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Raw data not found in {data_dir}")
        
    with open(text_path, 'r', encoding='utf-8') as f_t, open(label_path, 'r', encoding='utf-8') as f_l:
        texts = [line.strip() for line in f_t]
        labels = [LABEL_MAP[line.strip()] for line in f_l]
    
    return train_test_split(texts, labels, test_size=test_size, random_state=42)



def _print_data_distribution(labels: List[int], title: str = "Dataset"):
    """辅助函数：统计并打印数据分布"""
    
    if isinstance(labels, torch.Tensor) or isinstance(labels, np.ndarray):
        # numpy 与 torch 具有相同的接口, 且对后者而言, .tolist() 会自动处理 cpu/cuda 和 detach
        labels = labels.tolist()
        
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    logger.info(f"--- {title} Distribution (Total: {total_samples}) ---")
    # 按 label id 排序打印，看着整齐
    for label_id in sorted(label_counts.keys()):
        count = label_counts[label_id]
        class_name = ID2LABEL.get(label_id)
        
        percentage = (count / total_samples) * 100
        logger.info(f"{class_name:<12}: {count:5d} samples, {percentage:6.2f}%")
    
    
def _load_data(cfg: DataConfig) -> Tuple[Dataset, Dataset, Dict]:
    """根据配置加载 MalAPI 数据集"""
    data_dir: Path = Path(cfg.data_dir)
    cache_dir: Path = data_dir / "preprocessed" 
    cache_dir.mkdir(parents=True, exist_ok=True) 
    
    # 定义缓存文件路径
    cache_train_path = cache_dir / f"{cfg.dataset_name}-train.pkl"
    cache_test_path = cache_dir / f"{cfg.dataset_name}-test.pkl"
    cache_vocab_path = cache_dir / f"{cfg.dataset_name}-vocab.pkl"
    
    # 加载数据 (Cache 或 Raw)
    if cache_train_path.exists() and cache_test_path.exists() and cache_vocab_path.exists():
        logger.debug(f"Cache Hit! Loading dataset: {cfg.dataset_name}")        
        train_data, test_data, vocab = read_pickle(
            cache_train_path, 
            cache_test_path, 
            cache_vocab_path
        )
        
    else:
        logger.debug("Cache miss. Processing raw text data...")
        X_train, X_test, y_train, y_test = _load_raw_text_data(data_dir=data_dir, test_size=cfg.test_size)
        
        # 获取最小词频配置
        min_freq = 1
        if cfg.nlp_config and cfg.nlp_config.min_freq:
            min_freq = cfg.nlp_config.min_freq
            
        # 构建词表
        vocab = build_vocab(X_train, min_freq)
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        
        # 保存缓存
        write_pickle(
            (cache_train_path, train_data),
            (cache_test_path, test_data),
            (cache_vocab_path, vocab)
        )
        logger.debug("Cache saved successfully.")

    # [核心修改] 无论数据来源如何，都在此处统一打印分布
    all_labels = train_data[1] + test_data[1]
    _print_data_distribution(all_labels, title = "Full Dataset")
    _print_data_distribution(train_data[1], title = "Train Set")
    _print_data_distribution(test_data[1], title = "Test Set")
    
        
    if not cfg.nlp_config or not cfg.nlp_config.max_len:
         raise ValueError("Config Error: 'nlp_config.max_len' is required for text datasets.")
    
    max_len = cfg.nlp_config.max_len
    
    if cfg.dataset_name == "malapi2019":
        train_ds = MalAPITextDataset(train_data[0], train_data[1], vocab, max_len, text_pipeline=default_preprocess)
        test_ds = MalAPITextDataset(test_data[0], test_data[1], vocab, max_len, text_pipeline=default_preprocess)
    
    elif cfg.dataset_name == "malapi2019-gc":
        train_ds = MalAPITextDataset(train_data[0], train_data[1], vocab, max_len, text_pipeline=dedup_preprocess)
        test_ds = MalAPITextDataset(test_data[0], test_data[1], vocab, max_len, text_pipeline=dedup_preprocess)
  
    return train_ds, test_ds, vocab



def build_datamodule(ctx: ExpContext) -> Tuple[DataLoader, DataLoader, Dict]:
    """ 外部调用的唯一入口: 直接调用模块函数加载
        返回训练数据加载器、验证数据的加载器、词表大小
    """
    cfg = ctx.data_config
    train_ds, test_ds, vocab = _load_data(cfg)
    

    if vocab:
        vocab_size = len(vocab)
        cfg.nlp_config.vocab_size = vocab_size
        logger.debug(f"Auto-updating vocab_size to {vocab_size}")
        
    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=ctx.train_config.batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=ctx.train_config.batch_size)
    
    return train_loader, val_loader, vocab

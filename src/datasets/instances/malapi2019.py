from src.utils.logx import logger
from src.utils.text_preprocessing import build_vocab, default_preprocess
from src.schemas.context import ExpContext, DataConfig

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from pathlib import Path
import torch
import pickle



class MalAPITextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
        # 缓存一下特殊的索引，避免由 dict 查找带来的开销
        self.unk_idx = self.vocab.get('<unk>', 0)
        self.pad_idx = self.vocab.get('<pad>', 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词 (逻辑与词表构建保持一致 build_vocab)
        tokens = default_preprocess(text)
        
        # 转索引 (查不到的给 unk_idx)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # 截断与填充, 填充使用 pad_idx (1)
        if len(indices) >= self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.pad_idx] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)



def _load_raw_text_data(data_dir: Path = "data/malapi2019", test_size: float = 0.2) -> Tuple:
    """读取原始 txt 文件并分割"""
    text_path = data_dir / "all_analysis_data.txt"
    label_path = data_dir / "labels.txt"
    
    label_map = {'Spyware': 0, 'Downloader': 1, 'Trojan': 2, 'Worms': 3,  'Adware': 4, 'Dropper': 5, 'Virus': 6, 'Backdoor': 7}
    
    if not text_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Raw data not found in {data_dir}")

    with open(text_path, 'r', encoding='utf-8') as f_t, open(label_path, 'r', encoding='utf-8') as f_l:
        texts = [line.strip() for line in f_t]
        labels = [label_map[line.strip()] for line in f_l]
        
    return train_test_split(texts, labels, test_size=test_size, random_state=42)


def _load_data(cfg: DataConfig):
    """根据配置加载 MalAPI 数据集"""
    data_dir = Path(cfg.data_dir)
    cache_dir = data_dir / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True) 
    
    # 尝试读取缓存
    if (cache_dir / "train.pkl").exists() and (cache_dir / "vocab.pkl").exists():
        logger.debug("Loading text dataset from cache...")
        with open(cache_dir / "train.pkl", "rb") as f: train_data = pickle.load(f)
        with open(cache_dir / "test.pkl", "rb") as f:  test_data = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f: vocab = pickle.load(f)
        
    else:
        logger.debug("Processing raw text data...")
        X_train, X_test, y_train, y_test = _load_raw_text_data(data_dir)
        
        # 获取最小词频配置, 倘若没有设置默认设为 1
        min_freq = 1
        if cfg.nlp_config and cfg.nlp_config.min_freq:
            min_freq = cfg.nlp_config.min_freq

        # 构建词表
        vocab = build_vocab(X_train, min_freq)
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        
        # 保存缓存
        with open(cache_dir / "train.pkl", "wb") as f: pickle.dump(train_data, f)
        with open(cache_dir / "test.pkl", "wb") as f: pickle.dump(test_data, f)
        with open(cache_dir / "vocab.pkl", "wb") as f: pickle.dump(vocab, f)

    # 实例化 Dataset, 校验 max_len
    if not cfg.nlp_config or not cfg.nlp_config.max_len:
         raise ValueError("Config Error: 'nlp_config.max_len' is required for text datasets.")
    
    max_len = cfg.nlp_config.max_len
    
    train_ds = MalAPITextDataset(train_data[0], train_data[1], vocab, max_len)
    test_ds = MalAPITextDataset(test_data[0], test_data[1], vocab, max_len)
    
    return train_ds, test_ds, vocab



def build_datamodule(ctx: ExpContext) -> Tuple[DataLoader, DataLoader, int]:
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
    
    return train_loader, val_loader, vocab_size

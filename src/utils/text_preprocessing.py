from collections import Counter
from typing import List, Dict, Union
from itertools import groupby
import logging


# 词元化函数，将文本拆分为词元, 基于空格进行分词
def tokenize(text: str) -> List[str]:
    return text.split()


def identity_preprocess(text: str) -> str:
    return text

def default_preprocess(text: str) -> List[str]:
    return text.strip().lower().split()


def ngram_preprocess(text: str, n: int = 2) -> List[str]:
    """ 通常只用于机器学习模型, 因为 LSTM 理论等价于无限长的 n-gram,
        而在 TextCNN 之中，使用多个不同尺寸的卷积核，例如 kernel_sizes=[2, 3, 4, 5], 
        等价于同时提取 2-gram, 3-gram, 4-gram, 5-gram 特征
    """
    # 基础分词
    tokens = text.strip().lower().split()
    if n == 1:
        return tokens
    
    if len(tokens) < n:
        return tokens 
        
    # 生成 N-gram 列表, e.g. tokens = [A, B, C], n=2, 通过切片生成 zip([A,B,C], [B,C]) -> (A,B), (B,C)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    
    # 拼接成字符串 "A_B", "B_C", 经过这种处理之后, 单个句子的长度会变短, 但是整个词表的长度会大幅提升
    return ["_".join(gram) for gram in ngrams]


def dedup_preprocess(text: Union[List[str], str]) -> List[str]:
    """ 预处理：标准化 + 连续重复去重 (Folding), 支持输入字符串或字符串列表。
        Example:
            Input:  "Open Read Read Read Close Open" 
            Output: ["open", "read", "close", "open"]
            
            Input:  ["Open", "Read", "Read", "Read", "Close"]
            Output: ["open", "read", "close"]
    """
    # 基础清洗与分词, 若是字符串先做分割
    if isinstance(text, str):
        text = text.strip().lower().split()
        
    tokens = [t.lower() for t in text if isinstance(t, str)]
          
    if not tokens:
        return []

    # 核心逻辑：利用 groupby 去除连续重复
    # 其中 k 是组名(token), g 是分组迭代器。我们只需要 k
    deduped_tokens = [k for k, g in groupby(tokens)]
    return deduped_tokens


# 生成词表，返回词汇到索引的映射
def build_vocab(texts: List[str], min_freq: int = 1) -> Dict:
    """构建词表"""
    logging.debug(f"Building vocab with min_freq={min_freq}...")
    all_tokens = []
    for text in texts:
        all_tokens.extend(default_preprocess(text))
    
    token_counts = Counter(all_tokens)
    
    # 词表映射: word -> idx (0留给unk, 1留给pad)
    vocab = {
        word: idx + 2 
        for idx, (word, count) in enumerate(token_counts.items()) 
        if count >= min_freq
    }
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    logging.debug(f"Vocab size: {len(vocab)}")
    return vocab



# 将文本转为索引
def text_to_indices(text, vocab):
    tokens = default_preprocess(text)
    return [vocab.get(word, vocab['<unk>']) for word in tokens]


# 填充序列或截断
def pad_sequence(seq, max_len, padding_value=1):
    return seq[:max_len] if len(seq) > max_len else seq + [padding_value] * (max_len - len(seq))
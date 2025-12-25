from collections import Counter
from typing import List

import logging

# 词元化函数，将文本拆分为词元, 基于空格进行分词
def tokenize(text: str) -> List[str]:
    return text.split()


def default_preprocess(text: str) -> List[str]:
    return text.strip().lower().split()


# 生成词表，返回词汇到索引的映射
def build_vocab(texts: List[str], min_freq: int =1):
    """构建词表"""
    logging.info(f"Building vocab with min_freq={min_freq}...")
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
    logging.info(f"Vocab size: {len(vocab)}")
    return vocab



# 将文本转为索引
def text_to_indices(text, vocab):
    tokens = tokenize(default_preprocess(text))
    return [vocab.get(word, vocab['<unk>']) for word in tokens]


# 填充序列或截断
def pad_sequence(seq, max_len, padding_value=1):
    return seq[:max_len] if len(seq) > max_len else seq + [padding_value] * (max_len - len(seq))
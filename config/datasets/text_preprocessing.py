# text_preprocessing.py
import torch
from torch.utils.data import Dataset
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk import download


# 预处理文本，将其转换为小写
def preprocess_text(text):
    return text.lower()  # 转为小写

# 词元化函数，将文本拆分为词元
def tokenize(text):
    return text.split()  # 基于空格进行分词

# 生成词表，返回词汇到索引的映射
def build_vocab(corpus, min_freq=1):
    # 将所有单词分解并统计频率
    tokenized_texts = [tokenize(preprocess_text(text)) for text in corpus]
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    token_counts = Counter(all_tokens)
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(token_counts.items()) if count >= min_freq}
    
    # 添加两个特殊的词符：'<unk>'（未知词）和'<pad>'（填充）
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab

# 将文本转为索引
def text_to_indices(text, vocab):
    tokens = tokenize(preprocess_text(text))
    return [vocab.get(word, vocab['<unk>']) for word in tokens]

# 填充序列或截断
def pad_sequence(seq, max_len, padding_value=1):
    return seq[:max_len] if len(seq) > max_len else seq + [padding_value] * (max_len - len(seq))
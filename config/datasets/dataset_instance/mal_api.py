import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import io
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

class MalAPIEmbedDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # print(torch.tensor(self.embeddings[idx]).shape)
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])


# https://www.kaggle.com/datasets/focatak/malapi2019 
def load(text_path, labels_path, cache_dir='data/malapi2019/preprocessed', test_size=0.2, random_state=42):
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

    # 文件名列表
    file_names = ['train_texts.pkl', 'test_texts.pkl', 'train_labels.pkl', 'test_labels.pkl', 'vocab.pkl']
    cache_files_exist = all(os.path.exists(os.path.join(cache_dir, file)) for file in file_names)

    if cache_files_exist:
        logger.debug("📦 已有缓存，正在加载缓存数据集...")
        train_texts, test_texts, train_labels, test_labels, vocab = [io.read_pickle(os.path.join(cache_dir, file)) for file in file_names]
    else:
        logger.debug("📦 首次处理，正在缓存数据集...")
        with open(text_path, 'r', encoding='utf-8') as text_file,\
             open(labels_path, 'r', encoding='utf-8') as label_file:

            
            texts, labels = [], []
            for text, label in zip(text_file.readlines(), label_file.readlines()):
                # 处理文本（去除换行符等）
                texts.append(text.strip())  
                # 处理标签（去除换行符，并转换为数字）
                labels.append(label_map[label.strip()])  

        # 将数据分割成训练集和测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state)

        # 缓存词表
        vocab = build_vocab(train_texts)

        logger.debug(f"词表大小：{len(vocab)}")
        logger.debug(f"训练集文本标签对: {len(train_texts)}, 测试集文本标签对: {len(test_texts)}")

        # 使用循环保存所有数据到缓存
        os.makedirs(cache_dir, exist_ok=True)
        for file, data in zip(file_names, [train_texts, test_texts, train_labels, test_labels, vocab]):
            fname = os.path.join(cache_dir, file)
            io.write_pickle(fname, data)
            logger.debug(f"已将 {file} 缓存成功...")


    return train_texts, test_texts, train_labels, test_labels, vocab # train_dataset, test_dataset, vocab

def load_fgsmemb(cache_dir= "data/malapi2019/emb-feature/LSTMTextClassifier/advexam-fgsm/"): 
    file_names = ['train_adv_embeddings.pkl', 'test_adv_embeddings.pkl']
    cache_files_exist = all(os.path.exists(os.path.join(cache_dir, file)) for file in file_names)
    if cache_files_exist:
        logger.debug("📦 malapi_fgsmemd 已有缓存，正在加载缓存数据集...")
        train_data = io.read_pickle(os.path.join(cache_dir, file_names[0]))  # 读取训练集
        test_data = io.read_pickle(os.path.join(cache_dir, file_names[1]))  # 读取测试集
        train_texts, train_labels = train_data
        test_texts, test_labels = test_data

    return train_texts, test_texts, train_labels, test_labels

def load_pgdemb(cache_dir= "data/malapi2019/emb-feature/LSTMTextClassifier/advexam-pgd/"):
    file_names = ['train_adv_embeddings.pkl', 'test_adv_embeddings.pkl']
    cache_files_exist = all(os.path.exists(os.path.join(cache_dir, file)) for file in file_names)
    if cache_files_exist:
        logger.debug("📦 malapi_pgdemd 已有缓存，正在加载缓存数据集...")        
        train_data = io.read_pickle(os.path.join(cache_dir, file_names[0]))  # 读取训练集
        test_data = io.read_pickle(os.path.join(cache_dir, file_names[1]))  # 读取测试集
        train_texts, train_labels = train_data
        test_texts, test_labels = test_data
        
    return train_texts, test_texts, train_labels, test_labels
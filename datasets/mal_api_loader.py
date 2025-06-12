import os
from sklearn.model_selection import train_test_split
from datasets.text_preprocessing import MalAPITextDataset, build_vocab
import torch

def load_mal_api_data(text_file, labels_file, split = True, cache_dir='data/malapi2019/preprocessed', test_size=0.2, random_state=42):
    # 设置缓存文件路径
    train_path = os.path.join(cache_dir, 'train_dataset.pt')
    test_path = os.path.join(cache_dir, 'test_dataset.pt')
    vocab_path = os.path.join(cache_dir, 'vocab.pt')
    full_path = os.path.join(cache_dir, 'full_dataset.pt')

    # 如果 split=True 则加载分割的训练/测试数据
    if split:
        # 如果缓存文件都存在，则直接加载
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(vocab_path):
            print("🔁 正在从缓存加载训练/测试数据集...")
            train_dataset = torch.load(train_path)
            test_dataset = torch.load(test_path)
            vocab = torch.load(vocab_path)
            return train_dataset, test_dataset, vocab
    else:
        if os.path.exists(full_path) and os.path.exists(vocab_path):
            print("🔁 正在从缓存加载完整数据集...")
            full_dataset = torch.load(full_path)
            vocab = torch.load(vocab_path)
            return full_dataset, vocab

    # 否则，执行预处理
    print("📦 正在首次处理并缓存数据集...")
    # 读取文本和标签
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [label.strip() for label in f.readlines()]

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

    # 将标签从字符串转换为整数
    labels = [label_map[label] for label in labels]

    # 将数据分割成训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state)

    # 构建词表
    vocab = build_vocab(train_texts)
    # print("词表大小：", len(vocab))  # 278

    # 创建训练和测试数据集
    # 传递的是划分好的文本和标签，而不是文件路径
    train_dataset = MalAPITextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
    test_dataset = MalAPITextDataset(texts=test_texts, labels=test_labels, vocab=vocab)
    # 获取所有数据集
    full_dataset = MalAPITextDataset(texts, labels, vocab)

    # 创建缓存目录并保存
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(train_dataset, train_path)
    torch.save(test_dataset, test_path)
    torch.save(vocab, vocab_path)
    torch.save(full_dataset, full_path)

    if split:
        return train_dataset, test_dataset, vocab
    else: 
        return full_dataset, vocab

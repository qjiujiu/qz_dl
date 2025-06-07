import os
from sklearn.model_selection import train_test_split
from utils.text_preprocessing import MalAPITextDataset, build_vocab

def load_mal_api_data(text_file, labels_file, test_size=0.2, random_state=42):
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

    # 创建训练和测试数据集
    # 现在传递的是划分好的文本和标签，而不是文件路径
    train_dataset = MalAPITextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
    test_dataset = MalAPITextDataset(texts=test_texts, labels=test_labels, vocab=vocab)

    return train_dataset, test_dataset, vocab
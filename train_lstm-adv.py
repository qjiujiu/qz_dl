import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.logger import logger_initiate
from tqdm import tqdm
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_adversarial_dataset(emb_path, train_ratio=0.8):
    """
    从给定的对抗样本缓存中加载数据并划分训练与测试集
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"缓存文件 {emb_path} 不存在")

    emb_tensor, label_tensor = torch.load(emb_path)
    dataset = TensorDataset(emb_tensor, label_tensor)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset

def train_model_from_adversarial(config, adv_cache_path, tag='ADV'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取模型训练参数
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    max_len = config['max_len']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    dropout_prob = config['dropout_prob']
    checkpoint_path = config['checkpoint_path']

    # 日志器
    logger = logger_initiate(log_level=config.get('log_level', 'INFO'), is_console=True, is_file=True, is_colorful=True)

    # 加载对抗样本数据
    train_dataset, test_dataset = load_adversarial_dataset(adv_cache_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 构建模型（不使用 embedding 层）
    model = LSTMTextClassifier(
        vocab_size=0,  # placeholder
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_len=max_len
    ).to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward_from_embedding(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / len(tepoch))

        logger.info(f"[{tag}] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        # 测试精度
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward_from_embedding(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = correct / total
        logger.info(f"[{tag}] Test Accuracy: {acc * 100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"[{tag}] Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    # 使用 FGSM 训练示例：
    config_path = "config/lstm_fgsm_config.yaml"  # 改为你要使用的 config 路径
    adv_data_path = "data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt"  # 对抗样本路径
    config = load_config(config_path)
    train_model_from_adversarial(config, adv_data_path, tag='FGSM')  # 'FGSM' 可改为 'PGD'

    # 使用 PGD 训练示例：
    # config_path = "config/lstm_pgd_config.yaml"
    # adv_data_path = "data/malapi2019/emb-feature/advexam-pgd/pgd.pt"
    # config = load_config(config_path)
    # train_model_from_adversarial(config, adv_data_path, tag='PGD')  # 'FGSM' 可改为 'PGD'

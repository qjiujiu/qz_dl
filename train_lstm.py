# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.mal_api_loader import load_mal_api_data
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.logger import logger_initiate
from tqdm import tqdm
import os
import yaml

# 加载配置文件
def load_config(config_path="config/lstm_config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# FGSM 攻击
def fgsm_attack(model, emb_input, labels, epsilon, device):
    emb_input = emb_input.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)
    # torch.nn.LSTM 的 cuDNN 实现在 eval() 模式下 禁止反向传播
    # model.eval()
    
    # 设置为 train 模式以支持 RNN 反向传播
    model.train()
    # 临时禁用 dropout
    original_dropout_p = model.dropout.p
    model.dropout.p = 0.0
    
    output = model.forward_from_embedding(emb_input)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    adv_emb = emb_input + epsilon * emb_input.grad.sign()
    return adv_emb.detach()

# PGD 攻击
def pgd_attack(model, emb_input, labels, epsilon, alpha, iters, device):
    ori = emb_input.clone().detach().to(device)
    adv = ori.clone().detach().requires_grad_(True)
    labels = labels.to(device)
    # torch.nn.LSTM 的 cuDNN 实现在 eval() 模式下 禁止反向传播
    # model.eval()

    # 设置为 train 模式以支持 RNN 反向传播
    model.train()
    # 临时禁用 dropout
    original_dropout_p = model.dropout.p
    model.dropout.p = 0.0

    for _ in range(iters):
        model.zero_grad()
        output = model.forward_from_embedding(adv)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        eta = torch.clamp(adv - ori, -epsilon, epsilon)
        adv = torch.clamp(ori + eta, 0, 1).detach_().requires_grad_(True)
    return adv.detach()

# 缓存嵌入表示
def cache_embeddings(model, dataloader, device, cache_path='data/malapi2019/emb-feature/clean-exam'):
    clean_path = os.path.join(cache_path, 'clean_examples.pt')
    if os.path.exists(clean_path):
        return torch.load(clean_path)
    
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Caching embeddings"):
            texts = texts.to(device)
            labels = labels.to(device)
            emb = model.embedding(texts)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())

    cached_emb = torch.cat(all_embeddings)
    cached_labels = torch.cat(all_labels)

    os.makedirs(cache_path, exist_ok=True)
    torch.save((cached_emb, cached_labels), clean_path)

    return cached_emb, cached_labels

# 训练模型
def train_model(config):
    # 设置超参数
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    max_len = config['max_len']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    dropout_prob = config['dropout_prob']
    
    # 数据加载
    train_dataset, test_dataset, vocab = load_mal_api_data(config['train_data'], config['train_labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = LSTMTextClassifier(
        vocab_size=len(vocab), 
        embedding_dim=embedding_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        max_len=max_len
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 日志记录
    logger = logger_initiate(log_level=config.get('log_level', 'INFO'), is_console=True, is_file=True, is_colorful=True)

    # 训练模型
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # 使用 tqdm 显示训练进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for texts, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / len(tepoch))

        # 打印每轮的损失
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
        
        # 评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluating", unit="batch") as tepoch:
                for texts, labels in tepoch:
                    outputs = model(texts)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    tepoch.set_postfix(accuracy=correct / total * 100)

        accuracy = correct / total
        logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

        # 如果当前模型效果最好，则保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config['checkpoint_path'])
            logger.info(f"Model saved to {config['checkpoint_path']}")


# FGSM 样本生成
def generate_fgsm_examples(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['batch_size']
    epsilon = config['fgsm_epsilon']

    full_dataset, vocab = load_mal_api_data(config['train_data'], config['train_labels'], split=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    ).to(device)
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))

    logger = logger_initiate(log_level='INFO', is_console=True, is_file=True, is_colorful=True)
    cached_emb, cached_labels = cache_embeddings(model, full_loader, device)
    adv_fgsm = fgsm_attack(model, cached_emb, cached_labels, epsilon, device)
    os.makedirs('data/malapi2019/emb-feature/advexam-fgsm', exist_ok=True)
    torch.save((adv_fgsm.cpu(), cached_labels), 'data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt')
    logger.info("FGSM adversarial examples saved.")

# PGD 样本生成
def generate_pgd_examples(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['batch_size']
    epsilon = config['pgd_epsilon']
    alpha = config['pgd_alpha']
    pgd_iters = config['pgd_iters']

    full_dataset, vocab = load_mal_api_data(config['train_data'], config['train_labels'], split=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    ).to(device)
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))

    logger = logger_initiate(log_level='INFO', is_console=True, is_file=True, is_colorful=True)
    cached_emb, cached_labels = cache_embeddings(model, full_loader, device)
    adv_pgd = pgd_attack(model, cached_emb, cached_labels, epsilon, alpha, pgd_iters, device)
    os.makedirs('data/malapi2019/emb-feature/advexam-pgd', exist_ok=True)
    torch.save((adv_pgd.cpu(), cached_labels), 'data/malapi2019/emb-feature/advexam-pgd/pgd.pt')
    logger.info("PGD adversarial examples saved.")

if __name__ == "__main__":
    config = load_config()  # 加载配置
    # train_model(config)     # 训练模型
    generate_fgsm_examples(config)
    generate_pgd_examples(config)
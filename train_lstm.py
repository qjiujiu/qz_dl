# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn
from datasets.mal_api_loader import load_mal_api_data
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from torch.utils.data import DataLoader
from utils.logger import logger_initiate
from tqdm import tqdm

# 加载配置文件
def load_config(config_path="config/lstm_config.yaml"):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

if __name__ == "__main__":
    config = load_config()  # 加载配置
    train_model(config)     # 训练模型
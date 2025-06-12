import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cv.lenet import LeNet
from utils.logger import logger_initiate
from tqdm import tqdm
import yaml

# 加载配置
def load_config(config_path="config/lenet_config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 训练模型
def train_model(config):
    # 超参数
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    checkpoint_path = config['checkpoint_path']

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数、优化器
    model = LeNet(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 日志初始化
    logger = logger_initiate(log_level=config.get('log_level', 'INFO'), is_console=True, is_file=True, is_colorful=True)

    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / len(tepoch))

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluating", unit="batch") as tepoch:
                for images, labels in tepoch:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    tepoch.set_postfix(accuracy=correct / total * 100)

        accuracy = correct / total
        logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    config = load_config()
    train_model(config)

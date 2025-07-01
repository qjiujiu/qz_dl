# test_lstm.py
import torch
from torch.utils.data import DataLoader
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from config.datasets.dataset_instance import mal_api
from config.logger import logger_initiate
from tqdm import tqdm
from utils.get_config import load_config

# 测试模型
def test_model(config):
    # 设置超参数
    max_len = config['max_len']
    batch_size = config['batch_size']

    # 数据加载
    _, test_dataset, vocab = mal_api.load(config['train_data'], config['train_labels'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=max_len
    )

    # 加载训练好的模型权重
    checkpoint_path = config['checkpoint_path']
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # 设置为评估模式

    # 日志记录
    logger = logger_initiate(log_level=config.get('log_level', 'INFO'), is_console=True, is_file=True, is_colorful=True)

    # 评估模型
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tepoch:
            for texts, labels in tepoch:
                x = model.embed(texts)
                print(x.shape)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(accuracy=correct / total * 100)

    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    config = load_config("config/lstm_config.yaml")  # 加载配置
    test_model(config)      # 测试模型
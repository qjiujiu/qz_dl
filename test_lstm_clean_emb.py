import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.logger import logger_initiate
from tqdm import tqdm
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_on_clean_embedding(config, tag="Adv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size']
    max_len = config['max_len']
    checkpoint_path = config['checkpoint_path']

    # 加载干净样本的嵌入表示
    cache_path = 'data/malapi2019/emb-feature/clean-exam/clean_examples.pt'
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"未找到缓存文件：{cache_path}")
    clean_emb, clean_labels = torch.load(cache_path)
    clean_dataset = TensorDataset(clean_emb, clean_labels)

    # 划分测试集（只用 20% 进行测试）
    test_size = int(0.2 * len(clean_dataset))
    train_size = len(clean_dataset) - test_size
    _, test_dataset = random_split(clean_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型（vocab_size 设置为任意正数即可）
    model = LSTMTextClassifier(
        vocab_size=0,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=max_len
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件不存在：{checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    logger = logger_initiate(log_level='INFO', is_console=True, is_file=True, is_colorful=True)

    correct, total = 0, 0
    with torch.no_grad():
        with tqdm(test_loader, desc=f"[{tag}] Testing on Clean Embedding", unit="batch") as tepoch:
            for emb_batch, label_batch in tepoch:
                emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
                outputs = model.forward_from_embedding(emb_batch)
                _, predicted = torch.max(outputs, dim=1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
                tepoch.set_postfix(accuracy=correct / total * 100)

    acc = correct / total
    logger.info(f"[{tag}] Clean Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    # 测试 FGSM 对抗模型
    config_fgsm = load_config("config/lstm_fgsm_config.yaml")
    test_on_clean_embedding(config_fgsm, tag="FGSM")

    # 测试 PGD 对抗模型
    # config_pgd = load_config("config/lstm_pgd_config.yaml")
    # test_on_clean_embedding(config_pgd, tag="PGD")

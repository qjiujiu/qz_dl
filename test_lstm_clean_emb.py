# 用干净样本测试 FGSM对抗训练模型，PGD对抗训练模型，以及干净模型的性能
import torch
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from config.logger import logger_initiate
from tqdm import tqdm
import os
from utils.get_config import load_config
from datasets.clean_emb_loader import load_clean_embedding_dataset

def test_on_clean_embedding(config, tag="Adv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size']
    max_len = config['max_len']
    checkpoint_path = config['checkpoint_path']
    cache_path = 'data/malapi2019/emb-feature/clean-exam/clean_examples.pt'
    
    # 加载数据集
    _, test_loader = load_clean_embedding_dataset(cache_path, batch_size)

    # 初始化模型（vocab_size 设置为任意正数即可）
    model = LSTMTextClassifier(
        vocab_size=278,
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
                outputs = model.forward(emb_batch)
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
    config_pgd = load_config("config/lstm_pgd_config.yaml")
    test_on_clean_embedding(config_pgd, tag="PGD")

    # 测试 干净 模型
    config_clean = load_config("config/lstm_config.yaml")
    test_on_clean_embedding(config_clean, tag="Clean")

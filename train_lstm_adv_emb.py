import torch
import torch.nn as nn
import torch.optim as optim
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from config.logger import logger_initiate, logging
from tqdm import tqdm
from utils.get_config import load_config
from datasets.adv_emb_loader import load_adversarial_dataset

# 日志器
logger = logger_initiate(log_level=logging.INFO, is_console=True, is_file=True, is_colorful=True)


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
    vocab_size = config['vocab_size']
    checkpoint_path = config['checkpoint_path']


    # 加载对抗样本数据
    train_loader, test_loader = load_adversarial_dataset(adv_cache_path,batch_size=batch_size)

    # 构建模型（不使用 embedding 层）
    model = LSTMTextClassifier(
        vocab_size=vocab_size,  # 词表大小
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
                outputs = model.forward(inputs)
                logger.error(f"inputs = {inputs.shape}, outputs = {outputs.shape}, labels = {labels.shape}")
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
                outputs = model.forward(inputs)
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
    # config_path = "config/lstm_fgsm_config.yaml"  # 改为你要使用的 config 路径
    # adv_data_path = "data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt"  # 对抗样本路径
    # config = load_config(config_path)
    # train_model_from_adversarial(config, adv_data_path, tag='FGSM')  # 'FGSM' 可改为 'PGD'

    # # 使用 PGD 训练示例：
    # config_path = "config/lstm_pgd_config.yaml"
    # adv_data_path = "data/malapi2019/emb-feature/advexam-pgd/pgd.pt"
    # config = load_config(config_path)
    # train_model_from_adversarial(config, adv_data_path, tag='PGD')  # 'FGSM' 可改为 'PGD'

    # 使用 MiniLM生成的干净Embedding样本 作为输入，训练 原LSTM 模型
    config_path = "config/lstm_miniLM_emb.yaml"
    emb_data_path = "data/malapi2019/emb-MiniLM-L6/clean_examples.pt"
    config = load_config(config_path)
    train_model_from_adversarial(config, emb_data_path, tag='miniLM-clean')

import torch
import torch.nn as nn
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.get_config import load_config
from datasets.adv_emb_loader import load_adversarial_dataset
from utils.logger import logger_initiate


def evaluate_model_on_adversarial(config, adv_test_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 日志器
    logger = logger_initiate(log_level=config.get('log_level', 'INFO'), is_console=True, is_file=False, is_colorful=True)

    # 模型配置参数
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    max_len = config['max_len']
    batch_size = config['batch_size']
    checkpoint_path = config['checkpoint_path']

    # 加载模型
    model = LSTMTextClassifier(
        vocab_size=278,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_len=max_len
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 加载对抗样本测试集
    _, test_loader = load_adversarial_dataset(
        emb_path=adv_test_path,
        batch_size=batch_size
    )

    # 测试评估
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    logger.info(f"Test Accuracy on adversarial dataset: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    # FGSM 对抗模型在 FGSM 对抗样本上测试
    # config_path = "config/lstm_fgsm_config.yaml"
    # adv_test_path = "data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt"

    # config = load_config(config_path)
    # evaluate_model_on_adversarial(config, adv_test_path)

    # # PGD 对抗模型在 PGD 对抗样本上测试
    config_path = "config/lstm_pgd_config.yaml"
    adv_test_path = "data/malapi2019/emb-feature/advexam-pgd/pgd.pt"
    config = load_config(config_path)
    evaluate_model_on_adversarial(config, adv_test_path)

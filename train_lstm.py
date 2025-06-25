# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.mal_api_loader import load_mal_api_data
from models.nlp.lstm_text_classifier import LSTMTextClassifier

from tqdm import tqdm
from utils.get_config import load_config

from config.parser import NlpCfgParams
from config.logger import logger

logger.is_debug(True)

# 训练模型
def train_model(config):
    # 设置超参数
    cfg = NlpCfgParams(
        batch_size=8,
        epochs=30,
        lr=0.001,
        dropout_prob=0.5,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=8,
        max_len=200,
        vocab_size=278 #178
    )

    logger.debug(cfg)

    # 数据加载
    train_dataset, test_dataset, vocab = load_mal_api_data(config['train_data'], config['train_labels'])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)


    model = LSTMTextClassifier(vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, 
                                hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim, max_len=cfg.max_len)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.setup_ctx(cfg)\
         .setup_loss(criterion)\
         .setup_optimizer(optimizer)
    
    # model.train_one_epoch(train_loader, test_loader)
    model.train_multiple_epochs(train_loader, test_loader, epochs=cfg.epochs)

    

    

if __name__ == "__main__":
    config = load_config("config/lstm_config.yaml")  # 加载配置
    train_model(config)     # 训练模型
# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.mal_api_loader import load_mal_api_data
from models.nlp.lstm_text_classifier import LSTMTextClassifier

from config.params_parser.parser import ArgsParser
from config.logger import logger

logger.is_debug(True)
    


# python train_lstm.py --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    logger.debug(cfg)

    # 数据加载器模块直接返回两个 loader
    train_dataset, test_dataset, vocab = load_mal_api_data(
        text_file="data/malapi2019/all_analysis_data.txt", 
        labels_file="data/malapi2019/labels.txt"
    )

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
    
    model.train_multiple_epochs(train_loader, test_loader, epochs=cfg.epochs)
    
import os, sys
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim

from models.nlp.lstm_text_classifier import LSTMTextClassifier
from config.params_parser.parser import NlpCfgParams
from config.logger import logger

logger.is_debug(True)


if __name__ == '__main__':
    # 加载默认配置
    cfg = NlpCfgParams(
        embedding_dim=128,
        hidden_dim=256,
        output_dim=8,
        max_len=200,
        vocab_size=178,
        batch_size=8,
        epochs=30,
        lr=0.001,
        dropout_prob=0.5
    )

    logger.debug(cfg)

    # 创建模型
    model = LSTMTextClassifier(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        max_len=cfg.max_len,
    ).to(cfg.device)


    # 构造随机输入测试模型是否可运行文本数据
    dummy_text_input = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_len)).to(cfg.device)

    # 检测文本与嵌入层的形状    
    try:
        with torch.no_grad():
            embed = model.embed(dummy_text_input)
            preds = model(embed)
        print(f"\n✅ 模型前向传播测试成功！嵌入向量形状: {embed.shape}, 预测结果形状: {preds.shape}")

    except Exception as e:
        print(f"\n❌ 模型前向传播失败：", e)
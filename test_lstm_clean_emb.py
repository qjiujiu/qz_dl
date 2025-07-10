# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger

from utils.models import (
    pick_model, 
    pick_embedding_encoder
)


logger.is_debug(True)

""" 使用说明
    - 文本输入: 
        - chenzc
            python test_lstm_clean_emb.py -ec id --model lstm  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-1304-46c8ff3d_weights.pth
            
        python test_lstm_clean_emb.py -ec id --model lstm  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-0954-ff28631f_weights.pth

    - Embedding 输入
        - chenzc
            python test_lstm_clean_emb.py --only-embed --model lstm -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-1304-46c8ff3d_weights.pth
            
        python test_lstm_clean_emb.py --only-embed --model lstm -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-0954-ff28631f_weights.pth

    若想载入权重，可添加 --checkpoint-path (缩写 -cp) 参数
    若想跳过文本格式数据直接传输向量，可添加 --only-embed

    特别强调，如果直接使用预训练的 model embedding 模块产出的向量训练，必须传入预训练模型的权重路径
"""



if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    model = pick_model(cfg, cfg.checkpoint_path)
    logger.debug(
        f"模型结构: {model}"
        f"预测头层数: {cfg.L + 1}"
    )

    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi_cleanemb", 
        batch_size=cfg.batch_size, 
    )

    logger.debug(f"本轮训练的超参数设置: {cfg}")
    logger.debug(f"使用的训练数据规模: {data_resource}")    


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.setup_ctx(cfg)\
        .setup_loss(criterion)\
        .setup_optimizer(optimizer)

    
    # 如果开启向量模式，会通过 encoder 来将索引转为向量，否则会使用模型自带的嵌入层
    encoder = pick_embedding_encoder(cfg, cfg.load_path)
    logger.debug(f"当前引用的外部的 encoder: {encoder}")
    
    # 最后一轮评估的结果就是测试集上面跑出来的结果
    
    result =  model.evalution(
        dataloader = data_resource.test_loader, 
        encoder = encoder
    )

    logger.debug(result)
    
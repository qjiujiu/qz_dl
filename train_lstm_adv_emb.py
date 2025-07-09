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

# python train_lstm_adv_emb.py --only-embed --model lstm -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-1304-46c8ff3d_weights.pth
if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    model = pick_model(cfg, cfg.checkpoint_path)
    logger.debug(
        f"模型结构: {model}"
        f"预测头层数: {cfg.L + 1}"
    )

    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi_fgsmemb", 
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


    # 如果开启向量模式，会通过 encoder 来将索引转为向量，再把向量丢给 model
    # 假设用户没有指定任何的词嵌入模型来做 encoder，会默认使用恒等映射来做 encoder，从而跳过模型自带的嵌入层
    
    encoder = pick_embedding_encoder(cfg, cfg.load_path)
    logger.debug(
        f"是否开启 embedding 模式: {cfg.only_embed}\n"  
        f"当前使用外部 encoder: {encoder}"
    )
    
    # 最后一轮评估的结果就是测试集上面跑出来的结果
    model.train_multiple_epochs(
        loader = data_resource.train_loader, 
        val_loader = data_resource.test_loader, 
        epochs = cfg.epochs, 
        encoder = encoder
    )
    
# train_lstm.py
import torch.nn as nn
import torch.optim as optim

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger

from utils.models import (
    pick_model, 
    pick_embedding_encoder
)


logger.is_debug(True)

""" 使用说明
- lstm 模型
    - 文本输入
        默认嵌入: 
            python train_lstm.py --model lstm  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
        
        word2vec嵌入：
            python train_lstm.py --model lstm  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec word2vec -lp ./checkpoints/malapiwv.wordvectors

        fasttext嵌入：
            python train_lstm.py --model lstm  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec fasttext -lp ./checkpoints/malapift.wordvectors
    
    - Embedding 输入
        python train_lstm.py --only-embed --model lstm -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
        
- conv1d 模型
    - 文本输入
        默认嵌入: 
            python train_lstm.py --model conv1d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
        
        word2vec嵌入：
            python train_lstm.py --model conv1d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec word2vec -lp ./checkpoints/malapiwv.wordvectors

        fasttext嵌入：
            python train_lstm.py --model conv1d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec fasttext -lp ./checkpoints/malapift.wordvectors
    
- conv2d 模型
    - 文本输入
        默认嵌入: 
            python train_lstm.py --model conv2d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
        
        word2vec嵌入：
            python train_lstm.py --model conv2d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec word2vec -lp ./checkpoints/malapiwv.wordvectors

        fasttext嵌入：
            python train_lstm.py --model conv2d  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -ec fasttext -lp ./checkpoints/malapift.wordvectors
      

    其它说明: 
        1. 若想载入权重，可添加 --checkpoint-path (缩写 -cp) 参数 
        2. 若想跳过文本格式数据直接传输向量，可添加 --encoder (缩写 -ec) 参数，encoder也可以使用先前模型预训练的 embedding 参数，这种情况必须
           传入预训练模型的权重路径
        3. 若想切换数据集 可添加 --dataset 数据集选项: malapi_cleanemb, malapi_fgsmemb, malapi_pgdemb

    特别强调，如果直接使用预训练的 model embedding 模块产出的向量训练，必须传入预训练模型的权重路径
"""



if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    model = pick_model(cfg, cfg.checkpoint_path)
    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi", 
        batch_size=cfg.batch_size, 
    )

    logger.debug(f"超参数设置: {cfg}")
    logger.debug(f"模型结构: {model}")
    logger.debug(f"使用的训练数据规模: {data_resource}")    
    

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.setup_ctx(cfg)\
        .setup_loss(criterion)\
        .setup_optimizer(optimizer)


    # 如果开启向量模式，会通过 encoder 来将索引转为向量，再把向量丢给 model
    encoder = pick_embedding_encoder(cfg, cfg.load_path, vocab=data_resource.vocab)
    logger.debug(
        f"是否开启 embedding 模式: {cfg.only_embed}"     # 是否开启向量模式
        f"当前使用外部 encoder: {encoder}"               # 若不开启默认为空
    )
    

    # 训练模式，默认使用该模式，其最后一轮评估的结果就是测试集上面跑出来的结果
    if cfg.x == 1:         
        model.train_multiple_epochs(
            loader = data_resource.train_loader, 
            val_loader = data_resource.test_loader, 
            epochs = cfg.epochs, 
            encoder = encoder
        )
 
    # 测试模式
    elif cfg.x == 0:
        result =  model.evalution(
            dataloader = data_resource.test_loader, 
            encoder = encoder
        )
        logger.debug(f"模型权重来自: {cfg.checkpoint_path}")
        logger.debug(f"测试集评估结果: {result}")
    
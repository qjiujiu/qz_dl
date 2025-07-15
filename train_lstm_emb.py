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
    - 文本输入: 
        默认嵌入+fgsm
        训练: python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type fgsm
        测试: python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi -cp checkpoints/2025-07-14/LSTMTextAdvClassifier/20250714-1834-d731b949_weights.pth -x 0
        
        默认嵌入+pgd
        训练: python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type pgd
        测试: python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi -cp checkpoints/2025-07-14/LSTMTextAdvClassifier/20250714-2004-5d4baa9e_weights.pth -x 0

        word2vec嵌入+fgsm
        训练：python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type fgsm -ec word2vec -lp ./checkpoints/malapiwv.wordvectors
        测试：python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type fgsm -ec word2vec -lp ./checkpoints/malapiwv.wordvectors -cp checkpoints/2025-07-15/LSTMTextAdvClassifier/20250715-1527-e2532fc7_weights.pth -x 0

        word2vec嵌入+pgd
        训练：python train_lstm_emb.py --model lstm-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type pgd -ec word2vec -lp ./checkpoints/malapiwv.wordvectors

        fasttext嵌入+fgsm

        -ec fasttext -lp ./checkpoints/malapift.wordvectors
    
    - Embedding 输入
        训练: python train_lstm_emb.py -ec id --model lstm-adv -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi_fgsmemb
        测试: python train_lstm_emb.py -ec id --model lstm-adv -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi_fgsmemb -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-0954-ff28631f_weights.pth -x 0


    其它说明:  
        1. 若想载入权重，可添加 --checkpoint-path (缩写 -cp) 参数 
        2. 若想切换数据集 可添加 --dataset 数据集选项: malapi_cleanemb, malapi_fgsmemb, malapi_pgdemb
        3. 若想直接使用向量，可添加 --encoder (缩写 -ec) 参数，encoder也可以使用先前模型预训练的 embedding 参数，这种情况必须
           传入预训练模型的权重路径
"""




if __name__ == "__main__":
    cfg =  ArgsParser().create_adv_config()
    model = pick_model(cfg, cfg.checkpoint_path)
    data_resource = TextDataSrc.load_dataset(
        dataset_name=cfg.dataset, 
        batch_size=cfg.batch_size, 
    )

    logger.debug(f"超参数设置: {cfg}")
    logger.debug(f"模型结构: {model}, 预测头层数: {cfg.L+1}")
    logger.debug(f"使用的数据集{cfg.dataset}, 数据规模: {data_resource}")    
    

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    
    model.setup_ctx(cfg)\
        .setup_loss(criterion)\
        .setup_optimizer(optimizer)

    
    # 如果开启向量模式，会通过 encoder 来将索引转为向量，否则会使用模型自带的嵌入层
    encoder = pick_embedding_encoder(cfg, cfg.load_path, vocab=data_resource.vocab)
    logger.debug(f"当前引用的外部的 encoder: {encoder}, 权重来自: {cfg.load_path}")


    # 训练模式，默认使用该模式，其最后一轮评估的结果就是测试集上面跑出来的结果
    if cfg.x == 1:         
        model.train_multiple_epochs(
            loader = data_resource.train_loader, 
            val_loader = data_resource.test_loader, 
            epochs = cfg.epochs, 
            encoder = encoder,
            adv_type = cfg.adv_type
        )
 
    # 测试模式
    elif cfg.x == 0:
        result =  model.evalution(
            dataloader = data_resource.test_loader, 
            encoder = encoder
        )
        logger.debug(f"模型权重来自: {cfg.checkpoint_path}")
        logger.debug(f"测试集评估结果: {result}")
    
    
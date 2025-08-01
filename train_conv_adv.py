import torch
import torch.nn as nn
import torch.optim as optim

from ablation.focal_loss import FocalLoss
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
        训练: python train_conv_adv.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type fgsm
                
        默认嵌入+pgd
        训练: python train_conv_adv.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type pgd --pgd-iters 2
        
        默认嵌入+L2高斯噪声
        训练: python train_conv_adv.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type l2-gaus
        
        默认嵌入+L-inf高斯噪声
        训练: python train_conv_adv.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --adv-type linf-gaus
        
        默认嵌入+atten:mlp
            fgsm：        python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten mlp --adv-type fgsm
            pgd(iter=1):  python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten mlp --adv-type pgd --pgd-iters 1 (pgd的迭代次数需测试1-5次)
            L2高斯噪声:    python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten mlp --adv-type l2-gaus
            L-inf高斯噪声: python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten mlp --adv-type linf-gaus

        默认嵌入+atten:self
            fgsm：        python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self --adv-type fgsm
            pgd(iter=1):  python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self --adv-type pgd --pgd-iters 1 (pgd的迭代次数需测试1-5次)
            L2高斯噪声:    python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self --adv-type l2-gaus
            L-inf高斯噪声: python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self --adv-type linf-gaus
        
        默认嵌入+atten：pe
            fgsm：        python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten pe --adv-type fgsm
            pgd(iter=1):  python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten pe --adv-type pgd --pgd-iters 1 (pgd的迭代次数需测试1-5次)
            L2高斯噪声:    python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten pe --adv-type l2-gaus
            L-inf高斯噪声: python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten pe --adv-type linf-gaus
        
        默认嵌入+atten：self-pe
            fgsm：        python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self-pe --adv-type fgsm
            pgd(iter=1):  python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self-pe --adv-type pgd --pgd-iters 1 (pgd的迭代次数需测试1-5次)
            L2高斯噪声:    python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self-pe --adv-type l2-gaus
            L-inf高斯噪声: python train_lstm_emb.py --model conv1d-adv  -bs 8 -ep 30 --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 --dataset malapi --atten self-pe --adv-type linf-gaus
 

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
    

    # 超参数 alpha 向量 （类别频率取倒数，然后归一化）
    alpha = torch.tensor([
        0.12, 0.10, 0.10, 0.10, 0.26, 0.11, 0.10, 0.10
    ], dtype=torch.float32)
    
    # 定义损失函数和优化器
    criterion = FocalLoss(label_smoothing=0.1)
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
            adv_type = cfg.adv_type,
            pgd_iters = cfg.pgd_iters
        )
 
    # 测试模式
    elif cfg.x == 0:
        result =  model.evalution(
            dataloader = data_resource.test_loader, 
            encoder = encoder
        )
        logger.debug(f"模型权重来自: {cfg.checkpoint_path}")
        logger.debug(f"测试集评估结果: {result}")
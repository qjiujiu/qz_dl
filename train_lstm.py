# train_lstm.py
import torch
import torch.optim as optim
import torch.nn as nn

from models.nlp.conv_text import Conv1dTextClassifier
from models.nlp.lstm_text_classifier import LSTMTextClassifier

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger


logger.is_debug(True)

def pick_model(cfg):
    """ 根据 model_name 返回对应的模型实例
    :param model_name: 模型名称，如 'lstm', 'conv1d'
    :param cfg: 配置对象，包含所有超参数
    :return: 实例化的模型
    """
    if cfg.model == 'lstm':
        return LSTMTextClassifier(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            layers=cfg.L 
        )
    elif cfg.model == 'conv1d':
        return Conv1dTextClassifier(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim
        )
    
    raise ValueError(f"Unknown model name: {cfg.model}")



# python train_lstm.py --model lstm  --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi", 
        batch_size=cfg.batch_size
    )

    logger.debug(f"本轮训练的超参数设置: {cfg}")
    logger.debug(f"使用的训练数据规模: {data_resource}")    

    model = pick_model(cfg)
    
    logger.debug(f"模型结构: {model}")
    logger.debug(f"预测头层数: {cfg.L + 1}")
    

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.setup_ctx(cfg)\
        .setup_loss(criterion)\
        .setup_optimizer(optimizer)

    # 最后一轮评估的结果就是测试集上面跑出来的结果
    model.train_multiple_epochs(
        loader = data_resource.train_loader, 
        val_loader = data_resource.test_loader, 
        epochs = cfg.epochs
    )
    
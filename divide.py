import torch
import torch.nn as nn
import torch.optim as optim

from ablation.focal_loss import FocalLoss
from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger

# python divide.py --dataset malapi
if __name__ == "__main__":
    cfg =  ArgsParser().create_adv_config()
    data_resource = TextDataSrc.load_dataset(
        dataset_name=cfg.dataset, 
        batch_size=cfg.batch_size, 
    )

    # 图神经网络的效果是注意力机制的子集
    logger.debug(data_resource)

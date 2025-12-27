from src.utils.logx import logger
from src.schemas.context import TrainConfig
from src.schemas.base_enums import OptimizerType, SchedulerType, LossType

import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 



def build_loss_fn(cfg: TrainConfig) -> nn.Module:
    """ NOTE 构建损失函数, 关于损失函数, 模型无论是单标签多分类, 或是多标签多分类, 全都建议直接输出原生 logits 而不做任何变动, 
        只需要使用 torch 提供的损失函数即可, 这样模型的改动量最小, 是一个工程方面的最佳实践, 
        
        NOTE e.g. 假设一个 lstm 模型, 先前用于单标签多分类，现在改用 BCEWithLogitsLoss, 那么这个损失函数,
        其实已将 Sigmoid + BCELoss  并在一起计算, 并用 LogSumExp 技巧保证数值稳定性, 不需要修改模型
    """
    creators = {
        # 多分类 (单选)
        LossType.CROSS_ENTROPY: lambda: nn.CrossEntropyLoss(),
        
        # 多标签多标签, 或者二分类
        LossType.BCE_LOGITS: lambda: nn.BCEWithLogitsLoss(),
        
        # 暂时用 CE 占位，如果未来实现了 FocalLoss 类可以替换这里
        LossType.FOCAL_LOSS: lambda: nn.CrossEntropyLoss(),
    }
    
    creator_fn = creators.get(cfg.loss_fn)
    
    if creator_fn is None:
        logger.warning(f"Loss type '{cfg.loss_fn}' not defined, falling back to CrossEntropy")
        creator_fn = creators.get(LossType.CROSS_ENTROPY)
        
    return creator_fn()


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> optim.Optimizer:
    """
    根据 TrainConfig 构建优化器 (Lambda 映射版)
    """
    params = model.parameters()
    creators = {
        OptimizerType.ADAMW: lambda: optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay),
        OptimizerType.ADAM: lambda: optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay),
        OptimizerType.SGD: lambda: optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    }

    # 获取构建函数 (使用 get 处理未知类型)
    creator_fn = creators.get(cfg.optiz)
    
    # 兜底逻辑
    if creator_fn is None:
        logger.warning(f"Unknown optimizer '{cfg.optiz}', defaulting to Adam.")
        creator_fn = creators.get(OptimizerType.ADAM)
        
    return creator_fn()


def build_scheduler(cfg: TrainConfig, optimizer: optim.Optimizer) -> lr_scheduler._LRScheduler: 
    """
    根据 TrainConfig 构建调度器 (Lambda 映射版)
    """
    creators = {
        SchedulerType.COSINE: lambda: lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        ),
        SchedulerType.LINEAR: lambda: lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs))
        )
    }
    
    creator_fn = creators.get(cfg.sched)
    
    if creator_fn is None:
        logger.warning(f"Unknown scheduler '{cfg.sched}', defaulting to Linear.")
        creator_fn = creators.get(SchedulerType.LINEAR)

    return creator_fn()
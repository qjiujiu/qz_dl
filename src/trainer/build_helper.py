from src.utils.logx import logger
from src.schemas.context import TrainConfig
from src.schemas.base_enums import OptimizerType, SchedulerType, LossType

import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 

def build_loss_fn(cfg: TrainConfig) -> nn.Module:
    """
    构建损失函数
    """
    creators = {
        LossType.CROSS_ENTROPY: lambda: nn.CrossEntropyLoss(),
    }
    creator_fn = creators.get(cfg.loss_fn)
    
    if creator_fn is None:
        creator_fn = creators.get(LossType.CROSS_ENTROPY)
        logger.warning("Loss not define, falling back to CrossEntropy")
        
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
        creator_fn = creators.get(OptimizerType.ADAM)
        logger.warning(f"Unknown optimizer '{cfg.optiz}', defaulting to Adam.")
        
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
        creator_fn = creators.get(SchedulerType.LINEAR)
        logger.warning(f"Unknown scheduler '{cfg.sched}', defaulting to Linear.")

    return creator_fn()
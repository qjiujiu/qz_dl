from src.utils.logx import logger
from src.schemas.context import ExpContext

from .conv1d_classifier import TextSeqClassifier
from .conv2d_classifier import Seq2ImageClassifier
from .lstm_classifer import LSTMSeqClassifier
from .tcn_classifer import TCNSeqClassifier

import torch.nn as nn


def build_model(ctx: ExpContext) -> nn.Module:
    """
    模型构建工厂：根据 ExpContext 中的配置实例化对应的模型。
    """
    
    # 获取配置对象, 提取所有模型通用的公共参数
    cfg = ctx.network_config
    
    common_args = {
        "vocab_size": ctx.data_config.nlp_config.vocab_size,
        "embedding_dim": ctx.data_config.nlp_config.embedding_dim,
        "output_dim": cfg.num_classes, 
        "dropout": cfg.dropout_prob,
        "plugin_type": cfg.plugin_type
    }

    model_name = cfg.name
    logger.info(f"Building model: {model_name} with params: {common_args}")

    # 根据模型名称分发
    model_creators = {
        LSTMSeqClassifier.__name__: lambda: LSTMSeqClassifier(
            **common_args, 
            hidden_dim = 256, 
            layers = 1
        ), 
        
        TCNSeqClassifier.__name__: lambda: TCNSeqClassifier(
            **common_args, 
            tcn_channels = [64, 128, 256]
        ),
        
        TextSeqClassifier.__name__: lambda: TextSeqClassifier(
            **common_args, 
            hidden_dim = 256
        ), 
        
        Seq2ImageClassifier.__name__: lambda: Seq2ImageClassifier(
            **common_args, 
            hidden_dim = 256
        ),
    }
   
    model_name = cfg.name
    creator = model_creators.get(model_name)
   
    if creator is None:
        raise ValueError(f"Unknown model name: '{cfg.name}'. Supported: lstm, tcn, conv1d, conv2d")
    
    return creator()
    
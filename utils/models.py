from models.nlp.conv_text import Conv1dTextClassifier
from models.nlp.lstm_text_classifier import LSTMTextClassifier

from config.params_parser.params_template import (
    CommonCfgParams, 
    NlpCfgParams,
    CvCfgParams
)
from utils import io
from typing import Union


def pick_model(cfg: CommonCfgParams, load_path: str = None):
    """ 根据 model_name 返回对应的模型实例
    """
    model = None

    if cfg.model == 'lstm':
        model = LSTMTextClassifier(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            layers=cfg.L 
        )
    elif cfg.model == 'conv1d':
        model = Conv1dTextClassifier(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim
        )
    
    if model is None:
        raise ValueError(f"Unknown model name: {cfg.model}")
    
    if load_path is not None: 
        model = io.load_model_weights(model, load_path, device=cfg.device)
    
    return model


def pick_embedding_encoder(cfg: NlpCfgParams, load_path: str = None):
    """ 根据 encoder-name 返回对应的 embedding encoder 模型,
        需要注意，encoder 是不参与训练的，因此在返回之后必须冻结其参数
    """
    encoder = None

    if not cfg.only_embed:
        return encoder
    
    # 使用预训练模型原先的 embedding 模块来做转化
    if cfg.encoder == "default": 
        model = pick_model(cfg, load_path)
        encoder = model.embedding

        # 冻结参数：不参与梯度更新
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder
    
    return encoder
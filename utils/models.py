from models.nlp.conv_text import Conv1dTextClassifier
from models.nlp.lstm_text_classifier import LSTMTextClassifier

from config.params_parser.params_template import CommonCfgParams 
from utils import io
from typing import Union


def pick_model(cfg: CommonCfgParams):
    """ 根据 model_name 返回对应的模型实例
    :param model_name: 模型名称，如 'lstm', 'conv1d'
    :param cfg: 配置对象，包含所有超参数
    :return: 实例化的模型
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
    
    if cfg.load_path is not None: 
        model = io.load_model_weights(model, cfg.load_path, device=cfg.device)
    
    return model
    
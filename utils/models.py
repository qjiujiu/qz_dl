from models.nlp.conv_text import Conv1dTextClassifier
from models.nlp.lstm_text_classifier import LSTMTextClassifier

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